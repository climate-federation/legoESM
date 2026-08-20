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
session id: 019fdbb6-7f31-71d0-85e9-a94ed5020635
--------
user
ADVERSARIAL REVIEW of an uncommitted change. Prefer refutation over agreement; find defects, do not summarize.

THE CHANGE (see `git diff` plus the untracked file scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch):
1. scripts/bench/bench_voronoi_partition_methods.py gains an opt-in --schedule-cost layer and a --lloyd flag. New function schedule_cost_row() wraps the production legoesm.parallel.sharded_dynamics.spmd_schedule_cost and derives coloring_gap = n_rounds - max_degree, plus a 'recolor' vs 'ownership_only' label.
2. tests/bench/test_bench_voronoi_partition_methods.py gains tests, including a monkeypatched-stub test of the gap derivation.
3. A new sbatch scans s8/s9/s10 at 64/128 devices on the CPU 'shared' partition.

PURPOSE: decide whether MPAS halo-round count can still be cut by better COLOURING (cheap) or only by a new OWNERSHIP objective (expensive, 7-14 days). The gap is the discriminator.

MEASURED SO FAR (verify my reasoning, attack the conclusions):
- L2/L3/L4 x {geometric,sfc} x nd 2..16: gap == 0 on all 24, but ALL rows report production_strategy='allgather' (counterfactual).
- s6 lloyd=0: np8 rounds geometric/sfc/metis = 7/7/6; np16 = 13/10/10, gap == 0 on all six, cells/device 2560 at np16 so these ARE ppermute rows.

ATTACK SPECIFICALLY:
(a) Is coloring_gap actually a valid discriminator? max_degree is returned by _build_ppermute_schedule — read it and confirm it is a genuine LOWER BOUND on the round count of the graph being coloured, and that it is computed on the SAME graph the rounds are. If max_degree is, say, a max over a different entity set (cells vs edges) or excludes self/residual pairs, the gap is meaningless and my whole verdict collapses. This is the load-bearing claim.
(b) Vizing: for a general graph, proper EDGE colouring needs max_degree or max_degree+1. If this schedule is really an edge colouring, gap==0 means optimal and gap can never exceed 1 — which would make my 'recolor has room' branch nearly unreachable and the whole test a near-tautology. Is the ppermute schedule an edge colouring, a vertex colouring, or something else? Does that change what gap>0 would even mean?
(c) rounds == n_dev-1 at small nd in my L2-L4 sweep. Is that the complete-graph regime, i.e. is the measurement simply saturated and uninformative below some cells/device? If so, does s6@16 (rounds 13/10/10 < 15) actually escape it, or is it still partly saturated?
(d) The sbatch: arms 1-2 claim to VALIDATE against the reference census in spmd_schedule_cost's docstring (subdiv-8 sfc 12/14, metis 13/19, geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18). Check the census was actually produced at lloyd=0 and with method= passed the way I pass it (raw mesh, reorder_target defaulting to n_dev). If my invocation differs from what produced the census in ANY way, arm 1 is not a validation and I must not claim it is.
(e) schedule_cost_row deliberately does NOT catch exceptions; I argued the scorer's divisibility ValueError is unreachable because reorder_voronoi_for_sharding pads to the reorder target and I always let reorder_target default to n_dev. Verify that padding claim in voronoi_partition.py. If it is wrong, an arm dies mid-sweep.
(f) The metadata/provenance: does a lloyd=0 row carry enough to stop it being read back as a production SCVT receipt? Does --lloyd actually reach the mesh build, and does --halo-depth (MPI lane, 2) leak into the schedule scoring anywhere (it must not — SPMD depth is 3)?
(g) Anything in the tests that is vacuous or that passes for the wrong reason. Note I already caught and fixed one vacuous test (hardcoding coloring_gap=0 passed on every fast fixture because the real gap is always 0 there).

Cite file:line. Be blunt about what is wrong.
codex
I’ll trace the scorer and mesh/reordering paths end to end, then compare the batch invocations against the docstring census. I’ll focus on evidence that falsifies the proposed interpretation.
exec
/usr/bin/bash -lc "git status --short && git diff -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py && sed -n '1,260p' scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && rg -n -C 4 \"def spmd_schedule_cost|def _build_ppermute_schedule|ppermute_schedule|reference census|subdiv-8|subdiv-9|reorder_voronoi_for_sharding|halo-depth|lloyd\" legoesm scripts tests" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 2 in 4071ms:
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py
?? .physics-validator/flag_reachability_audit/
?? .physics-validator/homogeneous_ice_nucleation/
?? .physics-validator/scaling_campaign/codex_consult_hundreds_2026-08-07.md
?? .physics-validator/scaling_campaign/codex_review_schedule_cost_2026-08-07.md
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
index 6aa0f4a40..c970b0b0c 100644
--- a/scripts/bench/bench_voronoi_partition_methods.py
+++ b/scripts/bench/bench_voronoi_partition_methods.py
@@ -8,6 +8,35 @@ Scaling-audit item 8.  Two independent measurement layers:
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
+   The DECISIVE column is ``coloring_gap = n_rounds - max_degree``, and it
+   picks which fix is even worth costing:
+
+   * ``gap > 0`` -> the colourer's finite greedy-order search has not
+     reached the graph's own lower bound, so a BETTER COLOURING can still
+     cut rounds without touching ownership (cheap).
+   * ``gap == 0`` -> colouring is exhausted at this ownership; only a
+     partitioner with a NEW objective (minimize boundary max-degree, not
+     edge cut) can lower the round count (expensive).
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
@@ -23,6 +52,11 @@ exactly one rank; owner range valid) before any metric is recorded.
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
 
@@ -128,6 +162,57 @@ def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
                      f"expected one of {METHODS}")
 
 
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
+    Adds ``coloring_gap = n_rounds - max_degree`` (see the module docstring:
+    ``>0`` means recolouring still has room, ``0`` means only a new ownership
+    objective can help).
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
+        # The decisive column: which fix is worth costing at all.
+        "coloring_gap": gap,
+        "coloring_headroom": "recolor" if gap > 0 else "ownership_only",
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
@@ -138,6 +223,18 @@ def main() -> int:
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
     p.add_argument("--out", type=str,
                    default="results/a1/voronoi_partition_quality.json")
     args = p.parse_args()
@@ -154,7 +251,8 @@ def main() -> int:
     from legoesm.grids.voronoi import create_voronoi_mesh
     from legoesm.parallel.voronoi_partition import resolve_partition_method
 
-    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
+    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
+                               lloyd_iterations=args.lloyd)
     if max(rank_counts) > int(mesh.nCells):
         raise SystemExit(
             f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
@@ -185,6 +283,18 @@ def main() -> int:
                   f"mean={q['halo_cells_mean']:8.1f} | "
                   f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
                   f"nbrs max={q['neighbor_ranks_max']}")
+            if args.schedule_cost:
+                sc = schedule_cost_row(mesh, method, n_ranks)
+                row["schedule"] = sc
+                note = ("  [COUNTERFACTUAL: production auto-selects "
+                        "allgather here, no ppermute schedule]"
+                        if sc["production_strategy"] == "allgather" else "")
+                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
+                      f"rounds={sc['n_rounds']:3d} "
+                      f"max_degree={sc['max_degree']:3d} "
+                      f"gap={sc['coloring_gap']:+d} "
+                      f"-> {sc['coloring_headroom']} "
+                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
 
     payload = {
         "rows": rows,
@@ -204,7 +314,9 @@ def main() -> int:
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
diff --git a/tests/bench/test_bench_voronoi_partition_methods.py b/tests/bench/test_bench_voronoi_partition_methods.py
index 21f861fd7..e5f80abda 100644
--- a/tests/bench/test_bench_voronoi_partition_methods.py
+++ b/tests/bench/test_bench_voronoi_partition_methods.py
@@ -146,3 +146,103 @@ def test_halo_matches_runtime_partition():
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
+    "n_rounds, max_degree, gap, headroom",
+    [(14, 10, 4, "recolor"), (12, 12, 0, "ownership_only")])
+def test_coloring_gap_is_derived_not_assumed(
+        monkeypatch, n_rounds, max_degree, gap, headroom):
+    """The gap/headroom must be COMPUTED from the scorer's two fields.
+
+    Non-vacuity, the hard way: on every mesh small enough to test quickly the
+    real gap is 0 (measured L2/L3/L4 x geometric/sfc x nd 2-16 — the colourer
+    lands exactly on ``max_degree`` every time), so a real-mesh assertion
+    cannot tell a correct subtraction from a hardcoded ``0``; that exact
+    mutation passed the first version of this test.  Stubbing the production
+    scorer with a KNOWN non-zero gap is what makes the assertion able to
+    fail, and it also pins the branch that decides whether recolouring is
+    worth costing at all.
+    """
+    stub = {
+        "n_rounds": n_rounds, "max_degree": max_degree,
+        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
+        "resolved_method": "geometric", "halo_depth": 3,
+        "cells_per_device": 99_999, "production_strategy": "ppermute",
+    }
+    import legoesm.parallel.sharded_dynamics as sd
+    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
+
+    sc = mod.schedule_cost_row(object(), "geometric", 8)
+    assert sc["coloring_gap"] == gap
+    assert sc["coloring_headroom"] == headroom
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
+    # Mesh provenance: a lloyd=0 synthetic mesh must never be readable as a
+    # production SCVT receipt.
+    assert payload2["metadata"]["extra"]["lloyd_iterations"] == 50
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
# lower bound decides it:
#
#   coloring_gap = n_rounds - max_degree
#     > 0  -> the finite greedy-order search has NOT reached the graph's
#             lower bound; a better COLOURING cuts rounds without touching
#             ownership.  Cheap lever, do that first.
#     == 0 -> colouring is exhausted at this ownership; only a partitioner
#             that lowers max_degree can cut rounds.  Expensive lever.
#
# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
#     (method x n_dev) at the production working points.
# (b) CONFIRMS a cheap fix: gap > 0 at any production (ppermute) row.
#     REFUTES it: gap == 0 on every production row -> the 7-14 day
#     ownership build is the only remaining path, and is then justified.
# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
#     step.  It cannot be shrunk further onto small meshes: measured
#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
#     but EVERY one of those rows auto-selects the ALLGATHER strategy
#     (cells/device below the threshold), so production runs no ppermute
#     schedule there and the number is counterfactual.  Only meshes big
#     enough to keep cells/device above the threshold answer the question.
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

run_arm () {  # $1=level  $2=rank-counts  $3=label
  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
  "$PY" "$BENCH" \
      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
      --methods geometric,sfc,metis --schedule-cost \
      --out "$OUT/schedule_cost_s$1.json"
  echo "[scan] arm $3 exit=$?"
  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
}

# Arms 1-2: KNOWN answers (the docstring census) — instrument validation.
run_arm 8  64,128  "1/3 VALIDATION s8"
run_arm 9  64,128  "2/3 VALIDATION s9"
# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
run_arm 10 128     "3/3 UNKNOWN s10"

echo "SCAN_DONE"
rg: legoesm: No such file or directory (os error 2)
tests/bench/test_bench_voronoi_partition_methods.py-197-
tests/bench/test_bench_voronoi_partition_methods.py-198-
tests/bench/test_bench_voronoi_partition_methods.py-199-def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
tests/bench/test_bench_voronoi_partition_methods.py-200-    """The schedule is scored at the SPMD production halo depth, NOT this
tests/bench/test_bench_voronoi_partition_methods.py:201:    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
tests/bench/test_bench_voronoi_partition_methods.py-202-    a different graph and quietly report the wrong lane's cost."""
tests/bench/test_bench_voronoi_partition_methods.py-203-    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
tests/bench/test_bench_voronoi_partition_methods.py-204-
tests/bench/test_bench_voronoi_partition_methods.py-205-    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
--
tests/bench/test_bench_voronoi_partition_methods.py-242-    payload2 = json.loads(out2.read_text())
tests/bench/test_bench_voronoi_partition_methods.py-243-    row = payload2["rows"][0]
tests/bench/test_bench_voronoi_partition_methods.py-244-    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
tests/bench/test_bench_voronoi_partition_methods.py-245-    assert payload2["metadata"]["extra"]["schedule_cost"] is True
tests/bench/test_bench_voronoi_partition_methods.py:246:    # Mesh provenance: a lloyd=0 synthetic mesh must never be readable as a
tests/bench/test_bench_voronoi_partition_methods.py-247-    # production SCVT receipt.
tests/bench/test_bench_voronoi_partition_methods.py:248:    assert payload2["metadata"]["extra"]["lloyd_iterations"] == 50
--
tests/distributed/test_voronoi_mpi.py-80-
tests/distributed/test_voronoi_mpi.py-81-
tests/distributed/test_voronoi_mpi.py-82-@pytest.fixture(scope="module")
tests/distributed/test_voronoi_mpi.py-83-def mesh():
tests/distributed/test_voronoi_mpi.py:84:    return create_voronoi_mesh(subdivision_level=SUBDIVISION_LEVEL, lloyd_iterations=5)
tests/distributed/test_voronoi_mpi.py-85-
tests/distributed/test_voronoi_mpi.py-86-
tests/distributed/test_voronoi_mpi.py-87-@pytest.fixture
tests/distributed/test_voronoi_mpi.py-88-def sigma():
--
scripts/bench/bench_mpas_spmd_scaling.py-3-ppermute halo).
scripts/bench/bench_mpas_spmd_scaling.py-4-
scripts/bench/bench_mpas_spmd_scaling.py-5-The Voronoi twin of ``bench_atm_latlon_spmd_scaling.py`` (mirrored
scripts/bench/bench_mpas_spmd_scaling.py-6-flag-for-flag where the grids allow): the global mesh is REORDERED with
scripts/bench/bench_mpas_spmd_scaling.py:7:``reorder_voronoi_for_sharding`` (METIS/RCB/Hilbert-SFC cell partition, ghost-
scripts/bench/bench_mpas_spmd_scaling.py-8-padded to an even device split) so each device's contiguous ``P("device")``
scripts/bench/bench_mpas_spmd_scaling.py-9-shard is a spatially compact cell cluster, then the SSP-RK3 step exchanges
scripts/bench/bench_mpas_spmd_scaling.py-10-only the partition-boundary halo per stage via ``jax.lax.ppermute``.
scripts/bench/bench_mpas_spmd_scaling.py-11-
--
scripts/bench/bench_mpas_spmd_scaling.py-96-MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
scripts/bench/bench_mpas_spmd_scaling.py-97-
scripts/bench/bench_mpas_spmd_scaling.py-98-
scripts/bench/bench_mpas_spmd_scaling.py-99-def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
scripts/bench/bench_mpas_spmd_scaling.py:100:                          moist=False, lloyd_iterations=50):
scripts/bench/bench_mpas_spmd_scaling.py-101-    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
scripts/bench/bench_mpas_spmd_scaling.py-102-
scripts/bench/bench_mpas_spmd_scaling.py-103-    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
scripts/bench/bench_mpas_spmd_scaling.py-104-    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
--
scripts/bench/bench_mpas_spmd_scaling.py-114-    )
scripts/bench/bench_mpas_spmd_scaling.py-115-    from legoesm.grids.vertical import create_sigma_coordinate
scripts/bench/bench_mpas_spmd_scaling.py-116-    from legoesm.grids.voronoi import create_voronoi_mesh
scripts/bench/bench_mpas_spmd_scaling.py-117-    from legoesm.parallel.mesh import create_voronoi_device_mesh
scripts/bench/bench_mpas_spmd_scaling.py:118:    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
scripts/bench/bench_mpas_spmd_scaling.py-119-
scripts/bench/bench_mpas_spmd_scaling.py-120-    mesh = create_voronoi_mesh(subdivision_level=subdivision,
scripts/bench/bench_mpas_spmd_scaling.py:121:                               lloyd_iterations=lloyd_iterations)
scripts/bench/bench_mpas_spmd_scaling.py:122:    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
scripts/bench/bench_mpas_spmd_scaling.py-123-    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
scripts/bench/bench_mpas_spmd_scaling.py-124-        # Padding only guarantees divisibility for reorder_target.
scripts/bench/bench_mpas_spmd_scaling.py-125-        raise SystemExit(
scripts/bench/bench_mpas_spmd_scaling.py-126-            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
--
scripts/bench/bench_mpas_spmd_scaling.py-179-    p.add_argument("--subdivision", type=int, default=5,
scripts/bench/bench_mpas_spmd_scaling.py-180-                   help="icosahedral subdivision level L "
scripts/bench/bench_mpas_spmd_scaling.py-181-                        "(nCells = 10*4^L + 2 before ghost padding)")
scripts/bench/bench_mpas_spmd_scaling.py-182-    p.add_argument("--nlev", type=int, default=8)
scripts/bench/bench_mpas_spmd_scaling.py:183:    p.add_argument("--lloyd", type=int, default=50,
scripts/bench/bench_mpas_spmd_scaling.py-184-                   help="Lloyd relaxation iterations for the mesh. 50 = "
scripts/bench/bench_mpas_spmd_scaling.py-185-                        "production SCVT; 0 = labelled synthetic scaling "
scripts/bench/bench_mpas_spmd_scaling.py-186-                        "mesh (scaling receipts only, never physics — "
scripts/bench/bench_mpas_spmd_scaling.py-187-                        "must match the prewarmed cache key at subdiv>=9).")
--
scripts/bench/bench_mpas_spmd_scaling.py-312-            f"the ghost padding only guarantees divisibility for the "
scripts/bench/bench_mpas_spmd_scaling.py-313-            f"partition target.")
scripts/bench/bench_mpas_spmd_scaling.py-314-    mesh, model, s0, dev_config = build_model_and_state(
scripts/bench/bench_mpas_spmd_scaling.py-315-        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
scripts/bench/bench_mpas_spmd_scaling.py:316:        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
scripts/bench/bench_mpas_spmd_scaling.py-317-
scripts/bench/bench_mpas_spmd_scaling.py-318-    if args.multicontroller:
scripts/bench/bench_mpas_spmd_scaling.py-319-        # Every process computed the reorder independently — assert the
scripts/bench/bench_mpas_spmd_scaling.py-320-        # partitions agree before any collective uses the halo schedule.
--
scripts/bench/bench_mpas_spmd_scaling.py-497-        component="mpas_atm",
scripts/bench/bench_mpas_spmd_scaling.py-498-        subdivision=args.subdivision, n_devices=nd,
scripts/bench/bench_mpas_spmd_scaling.py-499-        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py-500-        partition_method=args.partition_method, physics=args.physics,
scripts/bench/bench_mpas_spmd_scaling.py:501:        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
scripts/bench/bench_mpas_spmd_scaling.py-502-        # a row without this field could pass as a production-SCVT receipt.
scripts/bench/bench_mpas_spmd_scaling.py:503:        lloyd_iterations=args.lloyd,
scripts/bench/bench_mpas_spmd_scaling.py-504-        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
scripts/bench/bench_mpas_spmd_scaling.py-505-        # saying "auto" would not reveal whether ppermute or allgather
scripts/bench/bench_mpas_spmd_scaling.py-506-        # was actually measured (codex M3c-2 MINOR).
scripts/bench/bench_mpas_spmd_scaling.py-507-        halo_strategy_requested=args.halo_strategy,
--
tests/parallel/test_spmd_schedule_cost.py-9-import pytest
tests/parallel/test_spmd_schedule_cost.py-10-
tests/parallel/test_spmd_schedule_cost.py-11-from legoesm.grids.voronoi import create_voronoi_mesh
tests/parallel/test_spmd_schedule_cost.py-12-from legoesm.parallel import sharded_dynamics as sd
tests/parallel/test_spmd_schedule_cost.py:13:from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
tests/parallel/test_spmd_schedule_cost.py-14-
tests/parallel/test_spmd_schedule_cost.py-15-
tests/parallel/test_spmd_schedule_cost.py-16-@pytest.fixture(scope="module")
tests/parallel/test_spmd_schedule_cost.py-17-def mesh():
--
tests/parallel/test_spmd_schedule_cost.py-62-def test_already_reordered_mesh_is_not_reordered_again(mesh):
tests/parallel/test_spmd_schedule_cost.py-63-    """Production holds an already-reordered mesh. Re-splitting it would score
tests/parallel/test_spmd_schedule_cost.py-64-    a mesh no run uses, so that path must be expressible and must agree with
tests/parallel/test_spmd_schedule_cost.py-65-    scoring the raw mesh once."""
tests/parallel/test_spmd_schedule_cost.py:66:    prepared = reorder_voronoi_for_sharding(mesh, 8, method="sfc")
tests/parallel/test_spmd_schedule_cost.py-67-    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
tests/parallel/test_spmd_schedule_cost.py-68-    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
tests/parallel/test_spmd_schedule_cost.py-69-    assert pre["resolved_method"] == "pre-reordered"
tests/parallel/test_spmd_schedule_cost.py-70-    assert pre["reorder_target"] is None, (
--
tests/parallel/test_spmd_schedule_cost.py-101-
tests/parallel/test_spmd_schedule_cost.py-102-@pytest.mark.slow
tests/parallel/test_spmd_schedule_cost.py-103-def test_sfc_beats_metis_and_geometric_on_rounds():
tests/parallel/test_spmd_schedule_cost.py-104-    """The finding this function exists to make measurable. Census at
tests/parallel/test_spmd_schedule_cost.py:105:    production sizes: subdiv-8 sfc 12/14 rounds at 64/128 devices vs metis
tests/parallel/test_spmd_schedule_cost.py-106-    13/19, geometric 16/21. Same ordering here at the SMALLEST size that can
tests/parallel/test_spmd_schedule_cost.py-107-    still tell the methods apart.
tests/parallel/test_spmd_schedule_cost.py-108-
tests/parallel/test_spmd_schedule_cost.py-109-    subdiv-4@8 and subdiv-5@8 score all three methods identically (7 rounds) —
--
tests/parallel/test_spmd_schedule_cost.py-129-    """Spy, not inference: an accidental second reorder could still produce a
tests/parallel/test_spmd_schedule_cost.py-130-    coincidentally equal round count, so assert the call never happens."""
tests/parallel/test_spmd_schedule_cost.py-131-    from legoesm.parallel import voronoi_partition as vp
tests/parallel/test_spmd_schedule_cost.py-132-
tests/parallel/test_spmd_schedule_cost.py:133:    prepared = reorder_voronoi_for_sharding(mesh, 8, method="sfc")
tests/parallel/test_spmd_schedule_cost.py-134-    calls = []
tests/parallel/test_spmd_schedule_cost.py-135-    monkeypatch.setattr(
tests/parallel/test_spmd_schedule_cost.py:136:        vp, "reorder_voronoi_for_sharding",
tests/parallel/test_spmd_schedule_cost.py-137-        lambda *a, **k: calls.append(1) or prepared)
tests/parallel/test_spmd_schedule_cost.py-138-    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
tests/parallel/test_spmd_schedule_cost.py-139-    assert not calls, "already_reordered=True must not reorder the mesh"
tests/parallel/test_spmd_schedule_cost.py-140-
--
tests/bench/test_bench_mpas_spmd_gates.py-55-    # Parser accepts the gate flags (argparse would SystemExit on unknowns).
tests/bench/test_bench_mpas_spmd_gates.py-56-    src = Path(_BENCH).read_text()
tests/bench/test_bench_mpas_spmd_gates.py-57-    for flag in ("--parity-gate", "--check-conservation", "--mass-rtol",
tests/bench/test_bench_mpas_spmd_gates.py-58-                 "--multicontroller", "--coordinator", "--partition-method",
tests/bench/test_bench_mpas_spmd_gates.py:59:                 "--reorder-for", "--lloyd"):
tests/bench/test_bench_mpas_spmd_gates.py-60-        assert flag in src
tests/bench/test_bench_mpas_spmd_gates.py-61-
tests/bench/test_bench_mpas_spmd_gates.py-62-
tests/bench/test_bench_mpas_spmd_gates.py:63:def test_lloyd_flag_reaches_the_mesh_builder(monkeypatch):
tests/bench/test_bench_mpas_spmd_gates.py:64:    """``--lloyd 0`` must select the synthetic scaling mesh, not lloyd=50.
tests/bench/test_bench_mpas_spmd_gates.py-65-
tests/bench/test_bench_mpas_spmd_gates.py-66-    Non-vacuous by construction: the sentinel records the kwarg
tests/bench/test_bench_mpas_spmd_gates.py-67-    ``create_voronoi_mesh`` actually receives, so dropping the plumbing
tests/bench/test_bench_mpas_spmd_gates.py-68-    (the state before this flag existed) makes the assertion fail rather
--
tests/bench/test_bench_mpas_spmd_gates.py-79-        raise RuntimeError("stop-after-mesh-request")
tests/bench/test_bench_mpas_spmd_gates.py-80-
tests/bench/test_bench_mpas_spmd_gates.py-81-    monkeypatch.setattr(voronoi, "create_voronoi_mesh", _spy)
tests/bench/test_bench_mpas_spmd_gates.py-82-    with pytest.raises(RuntimeError, match="stop-after-mesh-request"):
tests/bench/test_bench_mpas_spmd_gates.py:83:        mod.build_model_and_state(4, 4, 1, 1, "sfc", lloyd_iterations=0)
tests/bench/test_bench_mpas_spmd_gates.py-84-    assert seen["level"] == 4
tests/bench/test_bench_mpas_spmd_gates.py:85:    assert seen["lloyd_iterations"] == 0
tests/bench/test_bench_mpas_spmd_gates.py-86-
tests/bench/test_bench_mpas_spmd_gates.py-87-
tests/bench/test_bench_mpas_spmd_gates.py-88-def test_gather_voronoi_state_spmd_round_trip():
tests/bench/test_bench_mpas_spmd_gates.py-89-    """Direct exercise of the new gather: shard -> gather == original."""
--
tests/bench/test_bench_mpas_spmd_gates.py-96-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/bench/test_bench_mpas_spmd_gates.py-97-    from legoesm.parallel.mesh import create_voronoi_device_mesh, shard_pytree
tests/bench/test_bench_mpas_spmd_gates.py-98-    from legoesm.parallel.sharded_dynamics import gather_voronoi_state_spmd
tests/bench/test_bench_mpas_spmd_gates.py-99-    from legoesm.parallel.voronoi_partition import (
tests/bench/test_bench_mpas_spmd_gates.py:100:        reorder_voronoi_for_sharding,
tests/bench/test_bench_mpas_spmd_gates.py-101-    )
tests/bench/test_bench_mpas_spmd_gates.py-102-
tests/bench/test_bench_mpas_spmd_gates.py-103-    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
tests/bench/test_bench_mpas_spmd_gates.py-104-    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
tests/bench/test_bench_mpas_spmd_gates.py-105-
tests/bench/test_bench_mpas_spmd_gates.py:106:    mesh = reorder_voronoi_for_sharding(
tests/bench/test_bench_mpas_spmd_gates.py-107-        create_voronoi_mesh(subdivision_level=3), 2, method="sfc")
tests/bench/test_bench_mpas_spmd_gates.py-108-    state = baroclinic_wave_init_mpas(
tests/bench/test_bench_mpas_spmd_gates.py-109-        mesh, create_sigma_coordinate(4), perturbed=True)
tests/bench/test_bench_mpas_spmd_gates.py-110-    dev = create_voronoi_device_mesh(
--
tests/validation/test_cross_grid_rce_smoke.py-321-        compute_terrain_metric, create_height_coordinate,
tests/validation/test_cross_grid_rce_smoke.py-322-    )
tests/validation/test_cross_grid_rce_smoke.py-323-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/validation/test_cross_grid_rce_smoke.py-324-
tests/validation/test_cross_grid_rce_smoke.py:325:    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
tests/validation/test_cross_grid_rce_smoke.py-326-    hc = create_height_coordinate(NLEV, H_TOP)
tests/validation/test_cross_grid_rce_smoke.py-327-    tm = compute_terrain_metric(jnp.zeros(mesh.nCells), hc)
tests/validation/test_cross_grid_rce_smoke.py-328-    cfg = MPASCompressibleEulerConfig(
tests/validation/test_cross_grid_rce_smoke.py-329-        nu_del2=1.0e4, n_acoustic_substeps=4, fix_mass=False,
--
scripts/data/generate_mpas_nmc.py-77-def _make_model(config: dict):
scripts/data/generate_mpas_nmc.py-78-    mesh = create_grid(
scripts/data/generate_mpas_nmc.py-79-        "mpas",
scripts/data/generate_mpas_nmc.py-80-        int(config["resolution"]),
scripts/data/generate_mpas_nmc.py:81:        lloyd_iterations=int(config["lloyd_iterations"]),
scripts/data/generate_mpas_nmc.py-82-    )
scripts/data/generate_mpas_nmc.py-83-    levels = int(config["levels"])
scripts/data/generate_mpas_nmc.py-84-    if config["vertical_coordinate"] == "hybrid":
scripts/data/generate_mpas_nmc.py-85-        sigma = make_hybrid_levels(levels, p_top_Pa=200.0, stretching=2.0)
--
scripts/run/run_rce.py-301-        from legoesm.grids.latlon import create_latlon_grid
scripts/run/run_rce.py-302-        grid = create_latlon_grid(N)
scripts/run/run_rce.py-303-    elif grid_type == "voronoi":
scripts/run/run_rce.py-304-        from legoesm.grids.voronoi import create_voronoi_mesh
scripts/run/run_rce.py:305:        grid = create_voronoi_mesh(N, lloyd_iterations=50)
scripts/run/run_rce.py-306-    else:
scripts/run/run_rce.py-307-        raise ValueError(f"Unknown grid type: {grid_type}")
scripts/run/run_rce.py-308-
scripts/run/run_rce.py-309-    sigma = create_sigma_coordinate(NLEV)
--
scripts/plot/plot_scaling_paper_figure.py-35-                  "LL2048@64 26502539, @128 26534060, LL2304@96/144 26628072/26657279, fused 26681636/26681858",
scripts/plot/plot_scaling_paper_figure.py-36-    "atm_cube": "26452894/26453782",
scripts/plot/plot_scaling_paper_figure.py-37-    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
scripts/plot/plot_scaling_paper_figure.py-38-                "s8 np32-128 26549646/26538474, s9 26600095, "
scripts/plot/plot_scaling_paper_figure.py:39:                "s8-lloyd0 26628076, s10@128 26677812",
scripts/plot/plot_scaling_paper_figure.py-40-    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic; "
scripts/plot/plot_scaling_paper_figure.py-41-                   "lat-lon 2-D r512 26628073",
scripts/plot/plot_scaling_paper_figure.py-42-    "oc_latlon": "26460444-501/26460365/26493592, LL2304@96/128 26646038/26646039",
scripts/plot/plot_scaling_paper_figure.py-43-    "oc_tripole": "26493837/26493648",
--
scripts/plot/plot_scaling_paper_figure.py-63-                ("float32 (C384)", [(6, 15.44), (24, 8.81)])],
scripts/plot/plot_scaling_paper_figure.py-64-        note="f64 pending",
scripts/plot/plot_scaling_paper_figure.py-65-    ),
scripts/plot/plot_scaling_paper_figure.py-66-    dict(
scripts/plot/plot_scaling_paper_figure.py:67:        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9/10 L26 · A100 NCCL",
scripts/plot/plot_scaling_paper_figure.py:68:        series=[("f32 (s8 · lloyd-50)", [(2, 19.90), (4, 14.12), (8, 6.92),
scripts/plot/plot_scaling_paper_figure.py-69-                                         (16, 7.10), (32, 8.13), (64, 5.27),
scripts/plot/plot_scaling_paper_figure.py-70-                                         (128, 6.47)]),
scripts/plot/plot_scaling_paper_figure.py:71:                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
scripts/plot/plot_scaling_paper_figure.py:72:                ("f32 (s8 lloyd-0)", [(8, 6.58), (16, 6.43), (32, 7.29)]),
scripts/plot/plot_scaling_paper_figure.py:73:                ("f32 (s10 lloyd-0)", [(128, 18.20)]),
scripts/plot/plot_scaling_paper_figure.py:74:                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
scripts/plot/plot_scaling_paper_figure.py-75-        note="s10@128 = 15.0 GC/s (record);\nweak 4x-cost decelerates 1.90→1.46",
scripts/plot/plot_scaling_paper_figure.py-76-    ),
scripts/plot/plot_scaling_paper_figure.py-77-    dict(
scripts/plot/plot_scaling_paper_figure.py-78-        key="atm_ico_cpu", title="ico + lat-lon 2-D", sub="subdiv-7 / r512 L26 · Milan CPU–MPI",
--
scripts/plot/plot_scaling_paper_figure.py-105-        key="oc_mpas", title="MPAS Voronoi", sub="subdiv-7/8 · Milan CPU–MPI, 32 rpn",
scripts/plot/plot_scaling_paper_figure.py-106-        series=[("float64 (subdiv-7)", [(32, 190.22), (64, 147.65),
scripts/plot/plot_scaling_paper_figure.py-107-                                        (128, 102.93), (256, 65.71),
scripts/plot/plot_scaling_paper_figure.py-108-                                        (512, 63.83)]),
scripts/plot/plot_scaling_paper_figure.py:109:                ("float64 (subdiv-8)", [(32, 861.25), (64, 494.89),
scripts/plot/plot_scaling_paper_figure.py-110-                                        (128, 309.05), (256, 254.41),
scripts/plot/plot_scaling_paper_figure.py-111-                                        (512, 194.80)])],
scripts/plot/plot_scaling_paper_figure.py-112-        note="32 ranks/node fixed",
scripts/plot/plot_scaling_paper_figure.py-113-    ),
--
scripts/plot/plot_scaling_paper_figure.py-119-          "mixed (f64 store)": "#009E73",
scripts/plot/plot_scaling_paper_figure.py-120-          "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
scripts/plot/plot_scaling_paper_figure.py-121-          "float32 (LL2048)": "#009E73",
scripts/plot/plot_scaling_paper_figure.py-122-          "f64 lat-lon 2-D (r512)": "#CC79A7",
scripts/plot/plot_scaling_paper_figure.py:123:          "f32 (s8 lloyd-0)": "#009E73",
scripts/plot/plot_scaling_paper_figure.py:124:          "f32 (s10 lloyd-0)": "#000000",
scripts/plot/plot_scaling_paper_figure.py-125-          "f32 (LL2304)": "#CC79A7",
scripts/plot/plot_scaling_paper_figure.py-126-          "f32 LL2048 fused+ovl": "#000000",
scripts/plot/plot_scaling_paper_figure.py:127:          "f32 (s8 · lloyd-50)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
scripts/plot/plot_scaling_paper_figure.py:128:          "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00"}
scripts/plot/plot_scaling_paper_figure.py-129-MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
scripts/plot/plot_scaling_paper_figure.py-130-           "f32 · LL1536/2048 @64": "*",
scripts/plot/plot_scaling_paper_figure.py-131-           "float64 (packed)": "s",
scripts/plot/plot_scaling_paper_figure.py-132-           "float32 (C768)": "o", "float32 (C384)": "^",
scripts/plot/plot_scaling_paper_figure.py-133-           "float32 (LL2048)": "^",
scripts/plot/plot_scaling_paper_figure.py-134-           "f64 lat-lon 2-D (r512)": "D",
scripts/plot/plot_scaling_paper_figure.py:135:           "f32 (s8 lloyd-0)": "v",
scripts/plot/plot_scaling_paper_figure.py:136:           "f32 (s10 lloyd-0)": "*",
scripts/plot/plot_scaling_paper_figure.py-137-           "f32 (LL2304)": "^",
scripts/plot/plot_scaling_paper_figure.py-138-           "f32 LL2048 fused+ovl": "*",
scripts/plot/plot_scaling_paper_figure.py:139:           "f32 (s8 · lloyd-50)": "o", "float32 (subdiv-9)": "^",
scripts/plot/plot_scaling_paper_figure.py:140:           "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v"}
scripts/plot/plot_scaling_paper_figure.py-141-
scripts/plot/plot_scaling_paper_figure.py-142-
scripts/plot/plot_scaling_paper_figure.py-143-def _style():
scripts/plot/plot_scaling_paper_figure.py-144-    plt.rcParams.update({
--
tests/ocean/run_ocean_all_grids_matrix.py-247-
tests/ocean/run_ocean_all_grids_matrix.py-248-    parser.add_argument("--mpas-base-mesh-level", type=int, default=2)
tests/ocean/run_ocean_all_grids_matrix.py-249-    parser.add_argument("--mpas-dt", type=float, default=60.0)
tests/ocean/run_ocean_all_grids_matrix.py-250-    parser.add_argument("--save-every", type=int, default=10)
tests/ocean/run_ocean_all_grids_matrix.py:251:    parser.add_argument("--lloyd-iterations", type=int, default=30)
tests/ocean/run_ocean_all_grids_matrix.py-252-    parser.add_argument("--latlon-nlon", type=int, default=360)
tests/ocean/run_ocean_all_grids_matrix.py-253-    parser.add_argument("--latlon-nlat", type=int, default=181)
tests/ocean/run_ocean_all_grids_matrix.py-254-
tests/ocean/run_ocean_all_grids_matrix.py-255-    parser.add_argument("--days", type=float, default=5.0)
--
scripts/bench/bench_ocean_gpu_scaling.py-98-    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
scripts/bench/bench_ocean_gpu_scaling.py-99-    from legoesm.ocean.dynamics.ocean_model_mpas import (
scripts/bench/bench_ocean_gpu_scaling.py-100-        MPASOceanModel, MPASOceanConfig,
scripts/bench/bench_ocean_gpu_scaling.py-101-    )
scripts/bench/bench_ocean_gpu_scaling.py:102:    mesh = create_voronoi_mesh(subdivision_level=level, lloyd_iterations=5)
scripts/bench/bench_ocean_gpu_scaling.py-103-    z = create_ocean_z_star(n_levels=OCEAN_NLEV)
scripts/bench/bench_ocean_gpu_scaling.py-104-    cfg = MPASOceanConfig(barotropic_solver=baro_solver)
scripts/bench/bench_ocean_gpu_scaling.py-105-    model = MPASOceanModel(mesh, z, cfg)
scripts/bench/bench_ocean_gpu_scaling.py-106-    state = rest_state_mpas_ocean(mesh, z)
--
tests/parallel/test_ppermute_edge_coloring.py-1-"""Edge-coloring correctness + round-count for the route-B MPAS ppermute
tests/parallel/test_ppermute_edge_coloring.py:2:schedule (:func:`legoesm.parallel.sharded_dynamics._build_ppermute_schedule`).
tests/parallel/test_ppermute_edge_coloring.py-3-
tests/parallel/test_ppermute_edge_coloring.py-4-Each color is one bidirectional ppermute ROUND and the route-B lane is
tests/parallel/test_ppermute_edge_coloring.py-5-round-latency-bound (#1113), so the schedule builder picks the coloring
tests/parallel/test_ppermute_edge_coloring.py-6-with the fewest rounds across several deterministic first-fit orderings
--
tests/parallel/test_ppermute_edge_coloring.py-100-    pytest.importorskip("jax")
tests/parallel/test_ppermute_edge_coloring.py-101-
tests/parallel/test_ppermute_edge_coloring.py-102-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/parallel/test_ppermute_edge_coloring.py-103-    from legoesm.parallel.sharded_dynamics import (
tests/parallel/test_ppermute_edge_coloring.py:104:        _build_ppermute_schedule,
tests/parallel/test_ppermute_edge_coloring.py-105-        _build_voronoi_partition_infra,
tests/parallel/test_ppermute_edge_coloring.py-106-    )
tests/parallel/test_ppermute_edge_coloring.py-107-    from legoesm.parallel.voronoi_partition import (
tests/parallel/test_ppermute_edge_coloring.py:108:        reorder_voronoi_for_sharding,
tests/parallel/test_ppermute_edge_coloring.py-109-    )
tests/parallel/test_ppermute_edge_coloring.py-110-
tests/parallel/test_ppermute_edge_coloring.py-111-    n_dev = 16
tests/parallel/test_ppermute_edge_coloring.py-112-    mesh = create_voronoi_mesh(subdivision_level=3)
tests/parallel/test_ppermute_edge_coloring.py:113:    mesh = reorder_voronoi_for_sharding(mesh, n_dev, method="auto")
tests/parallel/test_ppermute_edge_coloring.py-114-    if mesh.nCells % n_dev or mesh.nEdges % n_dev:
tests/parallel/test_ppermute_edge_coloring.py-115-        pytest.skip("mesh not divisible by n_dev")
tests/parallel/test_ppermute_edge_coloring.py-116-    cells_per, edges_per = mesh.nCells // n_dev, mesh.nEdges // n_dev
tests/parallel/test_ppermute_edge_coloring.py-117-    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
tests/parallel/test_ppermute_edge_coloring.py-118-     cell_owner) = _build_voronoi_partition_infra(mesh, n_dev, halo_depth=3)
tests/parallel/test_ppermute_edge_coloring.py:119:    sched = _build_ppermute_schedule(
tests/parallel/test_ppermute_edge_coloring.py-120-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
tests/parallel/test_ppermute_edge_coloring.py-121-    )
tests/parallel/test_ppermute_edge_coloring.py-122-    # sorted greedy overshoots here (17); multi-start reaches the floor (14).
tests/parallel/test_ppermute_edge_coloring.py-123-    assert sched["n_rounds"] == sched["max_degree"], (
--
tests/distributed/test_voronoi_halo.py-45-
tests/distributed/test_voronoi_halo.py-46-@pytest.fixture(scope="module")
tests/distributed/test_voronoi_halo.py-47-def mesh():
tests/distributed/test_voronoi_halo.py-48-    """Small SCVT mesh for testing (162 cells)."""
tests/distributed/test_voronoi_halo.py:49:    return create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/distributed/test_voronoi_halo.py-50-
tests/distributed/test_voronoi_halo.py-51-
tests/distributed/test_voronoi_halo.py-52-# ========================================================================
tests/distributed/test_voronoi_halo.py-53-# Partition validity
--
tests/distributed/test_voronoi_halo.py-408-    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
tests/distributed/test_voronoi_halo.py-409-    def test_schedule_covers_all_halo_cells(self, mesh, n_ranks):
tests/distributed/test_voronoi_halo.py-410-        """Every halo cell is covered by exactly one ppermute round."""
tests/distributed/test_voronoi_halo.py-411-        from legoesm.parallel.voronoi_partition import (
tests/distributed/test_voronoi_halo.py:412:            reorder_voronoi_for_sharding,
tests/distributed/test_voronoi_halo.py-413-        )
tests/distributed/test_voronoi_halo.py-414-        from legoesm.parallel.sharded_dynamics import (
tests/distributed/test_voronoi_halo.py-415-            _build_voronoi_partition_infra,
tests/distributed/test_voronoi_halo.py:416:            _build_ppermute_schedule,
tests/distributed/test_voronoi_halo.py-417-        )
tests/distributed/test_voronoi_halo.py-418-
tests/distributed/test_voronoi_halo.py:419:        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
tests/distributed/test_voronoi_halo.py-420-        nCells = reordered.nCells
tests/distributed/test_voronoi_halo.py-421-        nEdges = reordered.nEdges
tests/distributed/test_voronoi_halo.py-422-        cells_per = nCells // n_ranks
tests/distributed/test_voronoi_halo.py-423-        edges_per = nEdges // n_ranks
tests/distributed/test_voronoi_halo.py-424-
tests/distributed/test_voronoi_halo.py-425-        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
tests/distributed/test_voronoi_halo.py-426-         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
tests/distributed/test_voronoi_halo.py-427-
tests/distributed/test_voronoi_halo.py:428:        sched = _build_ppermute_schedule(
tests/distributed/test_voronoi_halo.py-429-            partitions, cell_owner, n_ranks,
tests/distributed/test_voronoi_halo.py-430-            cells_per, edges_per, max_lc, max_le,
tests/distributed/test_voronoi_halo.py-431-        )
tests/distributed/test_voronoi_halo.py-432-
--
tests/distributed/test_voronoi_halo.py-454-    def test_ppermute_halo_matches_simulated_exchange(self, mesh, n_ranks):
tests/distributed/test_voronoi_halo.py-455-        """ppermute schedule produces same halo values as simulated
tests/distributed/test_voronoi_halo.py-456-        exchange."""
tests/distributed/test_voronoi_halo.py-457-        from legoesm.parallel.voronoi_partition import (
tests/distributed/test_voronoi_halo.py:458:            reorder_voronoi_for_sharding,
tests/distributed/test_voronoi_halo.py-459-        )
tests/distributed/test_voronoi_halo.py-460-        from legoesm.parallel.sharded_dynamics import (
tests/distributed/test_voronoi_halo.py-461-            _build_voronoi_partition_infra,
tests/distributed/test_voronoi_halo.py:462:            _build_ppermute_schedule,
tests/distributed/test_voronoi_halo.py-463-        )
tests/distributed/test_voronoi_halo.py-464-
tests/distributed/test_voronoi_halo.py:465:        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
tests/distributed/test_voronoi_halo.py-466-        nCells = reordered.nCells
tests/distributed/test_voronoi_halo.py-467-        cells_per = nCells // n_ranks
tests/distributed/test_voronoi_halo.py-468-        edges_per = reordered.nEdges // n_ranks
tests/distributed/test_voronoi_halo.py-469-
tests/distributed/test_voronoi_halo.py-470-        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
tests/distributed/test_voronoi_halo.py-471-         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
tests/distributed/test_voronoi_halo.py-472-
tests/distributed/test_voronoi_halo.py:473:        sched = _build_ppermute_schedule(
tests/distributed/test_voronoi_halo.py-474-            partitions, cell_owner, n_ranks,
tests/distributed/test_voronoi_halo.py-475-            cells_per, edges_per, max_lc, max_le,
tests/distributed/test_voronoi_halo.py-476-        )
tests/distributed/test_voronoi_halo.py-477-
--
tests/distributed/test_voronoi_halo.py-521-    @pytest.mark.parametrize("n_ranks", [2, 4])
tests/distributed/test_voronoi_halo.py-522-    def test_edge_coloring_valid(self, mesh, n_ranks):
tests/distributed/test_voronoi_halo.py-523-        """Each ppermute round has no device appearing as sender twice."""
tests/distributed/test_voronoi_halo.py-524-        from legoesm.parallel.voronoi_partition import (
tests/distributed/test_voronoi_halo.py:525:            reorder_voronoi_for_sharding,
tests/distributed/test_voronoi_halo.py-526-        )
tests/distributed/test_voronoi_halo.py-527-        from legoesm.parallel.sharded_dynamics import (
tests/distributed/test_voronoi_halo.py-528-            _build_voronoi_partition_infra,
tests/distributed/test_voronoi_halo.py:529:            _build_ppermute_schedule,
tests/distributed/test_voronoi_halo.py-530-        )
tests/distributed/test_voronoi_halo.py-531-
tests/distributed/test_voronoi_halo.py:532:        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
tests/distributed/test_voronoi_halo.py-533-        cells_per = reordered.nCells // n_ranks
tests/distributed/test_voronoi_halo.py-534-        edges_per = reordered.nEdges // n_ranks
tests/distributed/test_voronoi_halo.py-535-
tests/distributed/test_voronoi_halo.py-536-        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
tests/distributed/test_voronoi_halo.py-537-         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
tests/distributed/test_voronoi_halo.py-538-
tests/distributed/test_voronoi_halo.py:539:        sched = _build_ppermute_schedule(
tests/distributed/test_voronoi_halo.py-540-            partitions, cell_owner, n_ranks,
tests/distributed/test_voronoi_halo.py-541-            cells_per, edges_per, max_lc, max_le,
tests/distributed/test_voronoi_halo.py-542-        )
tests/distributed/test_voronoi_halo.py-543-
--
tests/test_cases/baroclinic_wave.py-798-    build" applies to the O(nCells*nlev) STATE leaves; O(nEntities) 1-D
tests/test_cases/baroclinic_wave.py-799-    temporaries (``cos(angleEdge)``) are still evaluated in full, like the
tests/test_cases/baroclinic_wave.py-800-    global mesh itself.
tests/test_cases/baroclinic_wave.py-801-
tests/test_cases/baroclinic_wave.py:802:    Requires ``reorder_voronoi_for_sharding`` padding (nCells and nEdges
tests/test_cases/baroclinic_wave.py-803-    divisible by the device count) — the same precondition
tests/test_cases/baroclinic_wave.py-804-    ``shard_pytree`` has.  ``dev_config.face_sharding is None`` (single
tests/test_cases/baroclinic_wave.py-805-    device) falls back to the global builder unchanged.
tests/test_cases/baroclinic_wave.py-806-    """
--
tests/validation/test_cross_grid_nh_consistency.py-244-        create_height_coordinate,
tests/validation/test_cross_grid_nh_consistency.py-245-    )
tests/validation/test_cross_grid_nh_consistency.py-246-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/validation/test_cross_grid_nh_consistency.py-247-
tests/validation/test_cross_grid_nh_consistency.py:248:    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
tests/validation/test_cross_grid_nh_consistency.py-249-    hc = create_height_coordinate(NLEV, H_TOP)
tests/validation/test_cross_grid_nh_consistency.py-250-    tm = compute_terrain_metric(
tests/validation/test_cross_grid_nh_consistency.py-251-        jnp.zeros(mesh.nCells, jnp.float64), hc,
tests/validation/test_cross_grid_nh_consistency.py-252-    )
--
tests/parallel/test_mpas_partitionlocal_build.py-45-        pytest.skip(f"needs {N_DEV} virtual devices "
tests/parallel/test_mpas_partitionlocal_build.py-46-                    f"(XLA_FLAGS did not take effect)")
tests/parallel/test_mpas_partitionlocal_build.py-47-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/parallel/test_mpas_partitionlocal_build.py-48-    from legoesm.parallel.mesh import create_voronoi_device_mesh
tests/parallel/test_mpas_partitionlocal_build.py:49:    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
tests/parallel/test_mpas_partitionlocal_build.py-50-
tests/parallel/test_mpas_partitionlocal_build.py-51-    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
tests/parallel/test_mpas_partitionlocal_build.py:52:    mesh = reorder_voronoi_for_sharding(mesh, N_DEV)
tests/parallel/test_mpas_partitionlocal_build.py-53-    assert mesh.nCells % N_DEV == 0 and mesh.nEdges % N_DEV == 0
tests/parallel/test_mpas_partitionlocal_build.py-54-    dev_config = create_voronoi_device_mesh(
tests/parallel/test_mpas_partitionlocal_build.py-55-        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
tests/parallel/test_mpas_partitionlocal_build.py-56-        n_devices=N_DEV,
--
scripts/run/mpas_3dvar_single/run_assimilation.py-55-    parser.add_argument("--gen-be", type=Path, default=Path(inputs["gen_be_path"]))
scripts/run/mpas_3dvar_single/run_assimilation.py-56-    parser.add_argument("--output-dir", type=Path, default=Path(config["output"]["directory"]))
scripts/run/mpas_3dvar_single/run_assimilation.py-57-    parser.add_argument("--resolution", type=int, default=int(grid["resolution"]))
scripts/run/mpas_3dvar_single/run_assimilation.py-58-    parser.add_argument("--nlev", type=int, default=int(grid["levels"]))
scripts/run/mpas_3dvar_single/run_assimilation.py:59:    parser.add_argument("--lloyd-iterations", type=int, default=int(grid["lloyd_iterations"]))
scripts/run/mpas_3dvar_single/run_assimilation.py-60-    parser.add_argument(
scripts/run/mpas_3dvar_single/run_assimilation.py-61-        "--background",
scripts/run/mpas_3dvar_single/run_assimilation.py-62-        choices=("f24", "f48"),
scripts/run/mpas_3dvar_single/run_assimilation.py-63-        default=str(config["background"]["sample_state"]),
--
scripts/run/mpas_3dvar_single/run_assimilation.py-396-        f"jax={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}",
scripts/run/mpas_3dvar_single/run_assimilation.py-397-        flush=True,
scripts/run/mpas_3dvar_single/run_assimilation.py-398-    )
scripts/run/mpas_3dvar_single/run_assimilation.py-399-
scripts/run/mpas_3dvar_single/run_assimilation.py:400:    mesh = create_grid("mpas", args.resolution, lloyd_iterations=args.lloyd_iterations)
scripts/run/mpas_3dvar_single/run_assimilation.py-401-    sigma = create_sigma_coordinate(args.nlev)
scripts/run/mpas_3dvar_single/run_assimilation.py-402-    background = _state_from_sample(args.sample, args.background)
scripts/run/mpas_3dvar_single/run_assimilation.py-403-    params = load_gen_be_params(args.gen_be)
scripts/run/mpas_3dvar_single/run_assimilation.py-404-    params = params._replace(len_scale=params.len_scale * float(args.len_scale_multiplier))
--
scripts/run/mpas_3dvar_single/run_assimilation.py-524-        "grid": {
scripts/run/mpas_3dvar_single/run_assimilation.py-525-            "type": "MPAS quasi-uniform",
scripts/run/mpas_3dvar_single/run_assimilation.py-526-            "resolution": args.resolution,
scripts/run/mpas_3dvar_single/run_assimilation.py-527-            "levels": args.nlev,
scripts/run/mpas_3dvar_single/run_assimilation.py:528:            "lloyd_iterations": args.lloyd_iterations,
scripts/run/mpas_3dvar_single/run_assimilation.py-529-        },
scripts/run/mpas_3dvar_single/run_assimilation.py-530-        "observation": {
scripts/run/mpas_3dvar_single/run_assimilation.py-531-            "variable": "temperature",
scripts/run/mpas_3dvar_single/run_assimilation.py-532-            "target_pressure_hpa": args.obs_pressure_hpa,
--
scripts/bench/run_levante_gpu_scaling.py-1550-        else:
scripts/bench/run_levante_gpu_scaling.py-1551-            # Single-node: reorder for spatial locality and shard.
scripts/bench/run_levante_gpu_scaling.py-1552-            if n_gpus > 1:
scripts/bench/run_levante_gpu_scaling.py-1553-                from legoesm.parallel.voronoi_partition import (
scripts/bench/run_levante_gpu_scaling.py:1554:                    reorder_voronoi_for_sharding,
scripts/bench/run_levante_gpu_scaling.py-1555-                )
scripts/bench/run_levante_gpu_scaling.py:1556:                grid = reorder_voronoi_for_sharding(grid, n_gpus)
scripts/bench/run_levante_gpu_scaling.py-1557-
scripts/bench/run_levante_gpu_scaling.py-1558-            dev_config = create_voronoi_device_mesh(
scripts/bench/run_levante_gpu_scaling.py-1559-                nCells=grid.nCells,
scripts/bench/run_levante_gpu_scaling.py-1560-                nEdges=grid.nEdges,
--
scripts/bench/bench_voronoi_partition_methods.py-28-   * ``gap == 0`` -> colouring is exhausted at this ownership; only a
scripts/bench/bench_voronoi_partition_methods.py-29-     partitioner with a NEW objective (minimize boundary max-degree, not
scripts/bench/bench_voronoi_partition_methods.py-30-     edge cut) can lower the round count (expensive).
scripts/bench/bench_voronoi_partition_methods.py-31-
scripts/bench/bench_voronoi_partition_methods.py:32:   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
scripts/bench/bench_voronoi_partition_methods.py-33-   flag is the MPI lane's (default 2), while the schedule is scored at the
scripts/bench/bench_voronoi_partition_methods.py-34-   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
scripts/bench/bench_voronoi_partition_methods.py-35-   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
scripts/bench/bench_voronoi_partition_methods.py-36-   evaluations of the integrator actually run.  When
--
scripts/bench/bench_voronoi_partition_methods.py-169-    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
scripts/bench/bench_voronoi_partition_methods.py-170-    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
scripts/bench/bench_voronoi_partition_methods.py-171-    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
scripts/bench/bench_voronoi_partition_methods.py-172-    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
scripts/bench/bench_voronoi_partition_methods.py:173:    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
scripts/bench/bench_voronoi_partition_methods.py-174-    lane's (2), and scoring the SPMD schedule at 2 would colour a different
scripts/bench/bench_voronoi_partition_methods.py-175-    graph.
scripts/bench/bench_voronoi_partition_methods.py-176-
scripts/bench/bench_voronoi_partition_methods.py-177-    Adds ``coloring_gap = n_rounds - max_degree`` (see the module docstring:
--
scripts/bench/bench_voronoi_partition_methods.py-180-
scripts/bench/bench_voronoi_partition_methods.py-181-    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
scripts/bench/bench_voronoi_partition_methods.py-182-    different reorder target, which would mis-slice the owned blocks — is
scripts/bench/bench_voronoi_partition_methods.py-183-    unreachable from here: this passes the raw mesh with the scorer's default
scripts/bench/bench_voronoi_partition_methods.py:184:    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
scripts/bench/bench_voronoi_partition_methods.py-185-    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
scripts/bench/bench_voronoi_partition_methods.py-186-    the call would therefore only swallow *unforeseen* failures into a row
scripts/bench/bench_voronoi_partition_methods.py-187-    that reads like an orderly skip, which is how a missing number turns into
scripts/bench/bench_voronoi_partition_methods.py-188-    a silently wrong table.  Rows already print as the sweep goes, so a raise
--
scripts/bench/bench_voronoi_partition_methods.py-219-        formatter_class=argparse.RawDescriptionHelpFormatter)
scripts/bench/bench_voronoi_partition_methods.py-220-    p.add_argument("--subdivision", type=int, default=5,
scripts/bench/bench_voronoi_partition_methods.py-221-                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
scripts/bench/bench_voronoi_partition_methods.py-222-    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
scripts/bench/bench_voronoi_partition_methods.py:223:    p.add_argument("--halo-depth", type=int, default=2,
scripts/bench/bench_voronoi_partition_methods.py-224-                   help="Halo layers (runtime default 2, del4 support).")
scripts/bench/bench_voronoi_partition_methods.py-225-    p.add_argument("--methods", type=str, default=",".join(METHODS))
scripts/bench/bench_voronoi_partition_methods.py-226-    p.add_argument("--schedule-cost", action="store_true",
scripts/bench/bench_voronoi_partition_methods.py-227-                   help="Also score the SPMD ppermute halo-schedule depth "
scripts/bench/bench_voronoi_partition_methods.py-228-                        "(n_rounds vs max_degree) per method x rank count. "
scripts/bench/bench_voronoi_partition_methods.py-229-                        "Uses the production SPMD halo depth, NOT "
scripts/bench/bench_voronoi_partition_methods.py:230:                        "--halo-depth. Expensive: minutes per candidate at "
scripts/bench/bench_voronoi_partition_methods.py-231-                        "subdiv>=8 — run it under batch.")
scripts/bench/bench_voronoi_partition_methods.py:232:    p.add_argument("--lloyd", type=int, default=50,
scripts/bench/bench_voronoi_partition_methods.py-233-                   help="Lloyd relaxation iterations for the mesh. 50 = the "
scripts/bench/bench_voronoi_partition_methods.py-234-                        "production SCVT key; 0 = the LABELLED synthetic "
scripts/bench/bench_voronoi_partition_methods.py:235:                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
scripts/bench/bench_voronoi_partition_methods.py-236-                        "masquerade as a production receipt, and it must "
scripts/bench/bench_voronoi_partition_methods.py-237-                        "match the prewarmed cache key at subdiv>=9.")
scripts/bench/bench_voronoi_partition_methods.py-238-    p.add_argument("--out", type=str,
scripts/bench/bench_voronoi_partition_methods.py-239-                   default="results/a1/voronoi_partition_quality.json")
--
scripts/bench/bench_voronoi_partition_methods.py-251-    from legoesm.grids.voronoi import create_voronoi_mesh
scripts/bench/bench_voronoi_partition_methods.py-252-    from legoesm.parallel.voronoi_partition import resolve_partition_method
scripts/bench/bench_voronoi_partition_methods.py-253-
scripts/bench/bench_voronoi_partition_methods.py-254-    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
scripts/bench/bench_voronoi_partition_methods.py:255:                               lloyd_iterations=args.lloyd)
scripts/bench/bench_voronoi_partition_methods.py-256-    if max(rank_counts) > int(mesh.nCells):
scripts/bench/bench_voronoi_partition_methods.py-257-        raise SystemExit(
scripts/bench/bench_voronoi_partition_methods.py-258-            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
scripts/bench/bench_voronoi_partition_methods.py-259-            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
--
scripts/bench/bench_voronoi_partition_methods.py-315-            transport="none",
scripts/bench/bench_voronoi_partition_methods.py-316-            extra={"rank_counts": rank_counts, "methods": methods,
scripts/bench/bench_voronoi_partition_methods.py-317-                   "halo_depth": args.halo_depth,
scripts/bench/bench_voronoi_partition_methods.py-318-                   "schedule_cost": bool(args.schedule_cost),
scripts/bench/bench_voronoi_partition_methods.py:319:                   "lloyd_iterations": args.lloyd},
scripts/bench/bench_voronoi_partition_methods.py-320-        )),
scripts/bench/bench_voronoi_partition_methods.py-321-    }
scripts/bench/bench_voronoi_partition_methods.py-322-    outdir = os.path.dirname(args.out)
scripts/bench/bench_voronoi_partition_methods.py-323-    if outdir:
--
scripts/bench/profile_mpas_ocean.py-1-"""Profile the MPAS-ocean I6/L20 step on a local GPU.
scripts/bench/profile_mpas_ocean.py-2-
scripts/bench/profile_mpas_ocean.py-3-Builds the MPAS Voronoi ocean model exactly as
scripts/bench/profile_mpas_ocean.py-4-``scripts/bench/bench_ocean_gpu_scaling.py`` does (subdivision_level=6,
scripts/bench/profile_mpas_ocean.py:5:lloyd_iterations=5, 20 z* levels, default ``MPASOceanConfig``), then
scripts/bench/profile_mpas_ocean.py-6-breaks per-step wall time into sub-stages using:
scripts/bench/profile_mpas_ocean.py-7-
scripts/bench/profile_mpas_ocean.py-8-* ``time.perf_counter()`` + ``jax.block_until_ready`` around isolated
scripts/bench/profile_mpas_ocean.py-9-  closures for each stage (full segment step via ``lax.scan``, bare
--
scripts/bench/profile_mpas_ocean.py-126-
scripts/bench/profile_mpas_ocean.py-127-def main() -> int:
scripts/bench/profile_mpas_ocean.py-128-    # --- I6/L20 baseline (matches bench_ocean_gpu_scaling). ----------------
scripts/bench/profile_mpas_ocean.py-129-    subdivision_level = 6
scripts/bench/profile_mpas_ocean.py:130:    lloyd_iterations = 5
scripts/bench/profile_mpas_ocean.py-131-    n_levels = 20
scripts/bench/profile_mpas_ocean.py-132-    dt = 600.0
scripts/bench/profile_mpas_ocean.py-133-    precision = "fp64" if os.environ.get("JAX_ENABLE_X64", "0") == "1" else "fp32"
scripts/bench/profile_mpas_ocean.py-134-
--
scripts/bench/profile_mpas_ocean.py-147-    # --- Build mesh + state + model ----------------------------------------
scripts/bench/profile_mpas_ocean.py-148-    t0 = time.perf_counter()
scripts/bench/profile_mpas_ocean.py-149-    mesh = create_voronoi_mesh(
scripts/bench/profile_mpas_ocean.py-150-        subdivision_level=subdivision_level,
scripts/bench/profile_mpas_ocean.py:151:        lloyd_iterations=lloyd_iterations,
scripts/bench/profile_mpas_ocean.py-152-    )
scripts/bench/profile_mpas_ocean.py-153-    z = create_ocean_z_star(n_levels=n_levels)
scripts/bench/profile_mpas_ocean.py-154-    cfg = MPASOceanConfig()
scripts/bench/profile_mpas_ocean.py-155-    model = MPASOceanModel(mesh, z, cfg)
--
scripts/run/run_omip_core2.py-1482-    return grid, z_coord, model, state, np.asarray(H_bathy)
scripts/run/run_omip_core2.py-1483-
scripts/run/run_omip_core2.py-1484-
scripts/run/run_omip_core2.py-1485-def build_mpas_ocean(nlev: int, H_max: float, mesh_path: str, level: int = 6,
scripts/run/run_omip_core2.py:1486:                     lloyd_iterations: int = 20, woa_init: bool = False,
scripts/run/run_omip_core2.py-1487-                     woa_t=None, woa_s=None, flat_bottom: bool = False,
scripts/run/run_omip_core2.py-1488-                     A_h=None, B_h=None, K_bih=None, C_smag_lap=None,
scripts/run/run_omip_core2.py-1489-                     pgf_scheme=None, tracer_advection=None, bottom_drag_r=None,
scripts/run/run_omip_core2.py-1490-                     bottom_drag_bbl_thickness=None, bottom_drag_bg_velocity=None,
--
scripts/run/run_omip_core2.py-3597-                        "the next increment).")
scripts/run/run_omip_core2.py-3598-    p.add_argument("--mpas-level", type=int, default=6,
scripts/run/run_omip_core2.py-3599-                   help="MPAS Voronoi subdivision level (nCells=10*4^level+2): "
scripts/run/run_omip_core2.py-3600-                        "5~230km, 6~115km (~ORCA1), 7~58km. For --grid mpas.")
scripts/run/run_omip_core2.py:3601:    p.add_argument("--mpas-lloyd", type=int, default=20,
scripts/run/run_omip_core2.py-3602-                   help="MPAS Lloyd-relaxation iterations at mesh build. For --grid mpas.")
scripts/run/run_omip_core2.py-3603-    p.add_argument("--cube-Ah", type=float, default=None,
scripts/run/run_omip_core2.py-3604-                   help="cube horizontal viscosity A_h override [m^2/s].")
scripts/run/run_omip_core2.py-3605-    p.add_argument("--cube-hyperdiff", type=float, default=None,
--
scripts/run/run_omip_core2.py-4888-        app_grid_type = "cubed_sphere"
scripts/run/run_omip_core2.py-4889-    elif args.grid == "mpas":
scripts/run/run_omip_core2.py-4890-        grid, z_coord, model, state, H_bathy = build_mpas_ocean(
scripts/run/run_omip_core2.py-4891-            args.nlev, args.H_max, args.mesh,
scripts/run/run_omip_core2.py:4892:            level=args.mpas_level, lloyd_iterations=args.mpas_lloyd,
scripts/run/run_omip_core2.py-4893-            woa_init=args.woa_init, woa_t=args.woa_t, woa_s=args.woa_s,
scripts/run/run_omip_core2.py-4894-            flat_bottom=args.flat_bottom, partial_cell=args.partial_cell,
scripts/run/run_omip_core2.py-4895-            freeze_floor=(True if args.freeze_floor else None),
scripts/run/run_omip_core2.py-4896-            freezing=_freezing_ovr,
--
tests/distributed/test_voronoi_batched_halo.py-56-
tests/distributed/test_voronoi_batched_halo.py-57-@pytest.fixture(scope="module")
tests/distributed/test_voronoi_batched_halo.py-58-def mesh():
tests/distributed/test_voronoi_batched_halo.py-59-    """Small SCVT mesh (162 cells) — same as test_voronoi_halo."""
tests/distributed/test_voronoi_batched_halo.py:60:    return create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/distributed/test_voronoi_batched_halo.py-61-
tests/distributed/test_voronoi_batched_halo.py-62-
tests/distributed/test_voronoi_batched_halo.py-63-def _build_all(mesh, n_ranks):
tests/distributed/test_voronoi_batched_halo.py-64-    """All ranks' partitions + batched schedules, in one process."""
--
tests/parallel/test_voronoi_sharded_equivalence.py-78-        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
tests/parallel/test_voronoi_sharded_equivalence.py-79-            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
tests/parallel/test_voronoi_sharded_equivalence.py-80-        )
tests/parallel/test_voronoi_sharded_equivalence.py-81-        from legoesm.parallel.voronoi_partition import (
tests/parallel/test_voronoi_sharded_equivalence.py:82:            reorder_voronoi_for_sharding,
tests/parallel/test_voronoi_sharded_equivalence.py-83-        )
tests/parallel/test_voronoi_sharded_equivalence.py-84-        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
tests/parallel/test_voronoi_sharded_equivalence.py-85-
tests/parallel/test_voronoi_sharded_equivalence.py-86-        # Lower dt at higher subdivision to stay CFL-stable (dx scales
--
tests/parallel/test_voronoi_sharded_equivalence.py-91-        # Reorder for the *target* device count on both single-device
tests/parallel/test_voronoi_sharded_equivalence.py-92-        # and multi-device paths so cell/edge indices match — direct
tests/parallel/test_voronoi_sharded_equivalence.py-93-        # array comparison is meaningful.
tests/parallel/test_voronoi_sharded_equivalence.py-94-        target = reorder_for if reorder_for is not None else max(devices, 2)
tests/parallel/test_voronoi_sharded_equivalence.py:95:        mesh = reorder_voronoi_for_sharding(mesh, target)
tests/parallel/test_voronoi_sharded_equivalence.py-96-        sigma = create_sigma_coordinate(n_lev)
tests/parallel/test_voronoi_sharded_equivalence.py-97-        cfg = MPASPrimitiveEquationConfig(
tests/parallel/test_voronoi_sharded_equivalence.py-98-            nu_del4=1e16, nu_del4_ps=1e16,
tests/parallel/test_voronoi_sharded_equivalence.py-99-            fix_mass=True, pv_scheme='energy',
--
tests/distributed/test_mpi_differentiability.py-342-            partition_cells_geometric,
tests/distributed/test_mpi_differentiability.py-343-            partition_voronoi_mesh,
tests/distributed/test_mpi_differentiability.py-344-        )
tests/distributed/test_mpi_differentiability.py-345-
tests/distributed/test_mpi_differentiability.py:346:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/distributed/test_mpi_differentiability.py-347-        n_ranks = MPI.COMM_WORLD.Get_size()
tests/distributed/test_mpi_differentiability.py-348-        cell_owner = partition_cells_geometric(mesh, n_ranks)
tests/distributed/test_mpi_differentiability.py-349-        parts = [
tests/distributed/test_mpi_differentiability.py-350-            partition_voronoi_mesh(mesh, n_ranks, r, cell_owner=cell_owner)
--
tests/unit/test_clubb_scheme.py-1071-    from legoesm.grids.vertical import create_sigma_coordinate
tests/unit/test_clubb_scheme.py-1072-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_clubb_scheme.py-1073-
tests/unit/test_clubb_scheme.py-1074-    nlev = 10
tests/unit/test_clubb_scheme.py:1075:    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
tests/unit/test_clubb_scheme.py-1076-    sigma = create_sigma_coordinate(nlev)
tests/unit/test_clubb_scheme.py-1077-    nC, nE = mesh.nCells, mesh.nEdges
tests/unit/test_clubb_scheme.py-1078-    # Sheared (2→12 m/s) + per-edge structured edge-normal wind so the Perot
tests/unit/test_clubb_scheme.py-1079-    # reconstruction yields non-zero, spatially-varying cell winds.
--
tests/unit/test_clubb_scheme.py-1152-    from legoesm.grids.vertical import create_sigma_coordinate
tests/unit/test_clubb_scheme.py-1153-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_clubb_scheme.py-1154-
tests/unit/test_clubb_scheme.py-1155-    nlev = 10
tests/unit/test_clubb_scheme.py:1156:    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
tests/unit/test_clubb_scheme.py-1157-    sigma = create_sigma_coordinate(nlev)
tests/unit/test_clubb_scheme.py-1158-    nC, nE = mesh.nCells, mesh.nEdges
tests/unit/test_clubb_scheme.py-1159-    rng = np.random.default_rng(0)
tests/unit/test_clubb_scheme.py-1160-    u_edge = jnp.asarray(np.linspace(2.0, 10.0, nlev)[None, :]
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-8-#SBATCH --exclusive
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-10-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-11-#SBATCH --output=mpas_s8_l0.%j.log
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:12:# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:14:# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:21:#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-22-#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-23-#             matched cells/GPU stay well above 1 (prior draft saw
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-24-#             1.80/1.35/1.41 on the CONFOUNDED pairs)
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-25-#   REFUTE  : ratios collapse toward ~1.0 -> the draft's "term" was the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-26-#             Lloyd-mesh confound, and MPAS-GPU weak scaling is near
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-27-#             ideal at matched tile.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:28:# Mesh: prewarmed into LEGOESM_MESH_CACHE_DIR (subdiv-8 is below the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-29-# big-mesh refuse threshold, so a cache miss falls back to in-process
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-30-# builds — slower, still correct).
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-31-set -uo pipefail
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-32-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-40-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-41-rc=0
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-42-for NP in 8 16 32; do
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-43-  NODES=$(( NP / 4 ))
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:44:  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-45-  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-46-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-47-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-48-      --multicontroller --n-devices "$NP" \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:49:      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-50-      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-53-echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
--
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-40-#     schedule there and the number is counterfactual.  Only meshes big
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-41-#     enough to keep cells/device above the threshold answer the question.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-42-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-43-# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:44:# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:45:#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:46:#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-47-# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-48-# A scan that misses the known answer is a broken instrument, and its s10
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-49-# row must not be believed.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-50-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:51:# lloyd=0 throughout: it is what the reference census used AND the cache key
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-52-# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-53-# synthetic scaling mesh, recorded in every row's metadata so it can never be
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-54-# read back as a production SCVT receipt.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-55-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-56-# Arms run cheapest-first and each writes its own JSON, so a later arm that
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-57-# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-58-# last and is the one genuinely at risk: the scorer is known not to have
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:59:# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-60-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-61-# SUBMIT (from the repo root):
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-62-#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-63-# ===========================================================================
--
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-82-"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-83-  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-84-
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-85-run_arm () {  # $1=level  $2=rank-counts  $3=label
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:86:  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-87-  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-88-  "$PY" "$BENCH" \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:89:      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-90-      --methods geometric,sfc,metis --schedule-cost \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-91-      --out "$OUT/schedule_cost_s$1.json"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-92-  echo "[scan] arm $3 exit=$?"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-93-  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
--
scripts/data/fit_mpas_gen_be.py-145-    errors = [_state_from_sample(path) for path in paths]
scripts/data/fit_mpas_gen_be.py-146-    mesh = create_grid(
scripts/data/fit_mpas_gen_be.py-147-        "mpas",
scripts/data/fit_mpas_gen_be.py-148-        int(model_config["resolution"]),
scripts/data/fit_mpas_gen_be.py:149:        lloyd_iterations=int(model_config["lloyd_iterations"]),
scripts/data/fit_mpas_gen_be.py-150-    )
scripts/data/fit_mpas_gen_be.py-151-    sigma = create_sigma_coordinate(int(model_config["levels"]))
scripts/data/fit_mpas_gen_be.py-152-    if int(mesh.grid_n_columns) != int(errors[0].p_s.data.shape[0]):
scripts/data/fit_mpas_gen_be.py-153-        raise ValueError("Configured MPAS grid does not match the NMC sample grid")
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-8-#SBATCH --exclusive
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-10-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-11-#SBATCH --output=mpas_s10_128.%j.log
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:12:# HUNDREDS-OF-GPUS MPAS, rung 1: subdiv-10 (10.49M cells, lloyd=0
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-13-# synthetic, prewarmed job 26628074) at 128 GPUs = 81.9k cells/GPU —
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-14-# the SAME near-matched tile as s8@np8 (6.58 ms) and s9@np32 (12.47 ms),
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-15-# extending the 4x-devices weak series a third rung.
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-16-# Falsifiability, written BEFORE submit:
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-32-JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-33-    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-34-  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-35-    --multicontroller --n-devices 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:36:    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-37-    --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-38-    --out "$OUTDIR/np128.jsonl" || { echo "np128 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-39-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-40-import json,math,sys
--
scripts/data/prewarm_voronoi_mesh.py-3-Levels above the routine cap (subdiv > 8) are cache-or-prewarm only
scripts/data/prewarm_voronoi_mesh.py-4-(``voronoi.py`` big-mesh policy, codex round-19): an MPI launch REFUSES a
scripts/data/prewarm_voronoi_mesh.py-5-cache miss so that N ranks can never each rebuild a multi-hour mesh. This
scripts/data/prewarm_voronoi_mesh.py-6-script is the designated single builder — run it ONCE per (level,
scripts/data/prewarm_voronoi_mesh.py:7:lloyd_iterations) on one process, then launch the parallel job.
scripts/data/prewarm_voronoi_mesh.py-8-
scripts/data/prewarm_voronoi_mesh.py-9-Mesh flavours
scripts/data/prewarm_voronoi_mesh.py-10--------------
scripts/data/prewarm_voronoi_mesh.py:11:``--lloyd 50`` (default) is the production SCVT. ``--lloyd 0`` is the
scripts/data/prewarm_voronoi_mesh.py-12-LABELLED SYNTHETIC SCALING MESH — a bisected icosahedron whose Voronoi
scripts/data/prewarm_voronoi_mesh.py-13-dual is valid for TRiSK but under-relaxed: measured at subdiv-6, area CV
scripts/data/prewarm_voronoi_mesh.py:14:0.084 vs 0.061 and 128-part imbalance 1.148 vs 1.095 against lloyd=50
scripts/data/prewarm_voronoi_mesh.py-15-(within the codex round-19 comparability gate). Use it for scaling
scripts/data/prewarm_voronoi_mesh.py-16-receipts, never for physics claims.
scripts/data/prewarm_voronoi_mesh.py-17-
scripts/data/prewarm_voronoi_mesh.py-18-Cache location: $LEGOESM_MESH_CACHE_DIR — put it on PROJECT /work space
--
scripts/data/prewarm_voronoi_mesh.py-20-
scripts/data/prewarm_voronoi_mesh.py-21-Usage
scripts/data/prewarm_voronoi_mesh.py-22------
scripts/data/prewarm_voronoi_mesh.py-23-    LEGOESM_MESH_CACHE_DIR=/work/.../mesh_cache \\
scripts/data/prewarm_voronoi_mesh.py:24:    python scripts/data/prewarm_voronoi_mesh.py --level 9 --lloyd 0
scripts/data/prewarm_voronoi_mesh.py-25-"""
scripts/data/prewarm_voronoi_mesh.py-26-from __future__ import annotations
scripts/data/prewarm_voronoi_mesh.py-27-
scripts/data/prewarm_voronoi_mesh.py-28-import argparse
--
scripts/data/prewarm_voronoi_mesh.py-34-    ap = argparse.ArgumentParser(description=__doc__)
scripts/data/prewarm_voronoi_mesh.py-35-    ap.add_argument("--level", type=int, required=True,
scripts/data/prewarm_voronoi_mesh.py-36-                    help="Icosahedral subdivision level (9 or 10 need this "
scripts/data/prewarm_voronoi_mesh.py-37-                         "script; <=8 build routinely without it).")
scripts/data/prewarm_voronoi_mesh.py:38:    ap.add_argument("--lloyd", type=int, default=50,
scripts/data/prewarm_voronoi_mesh.py-39-                    help="Lloyd iterations: 50 = production SCVT, 0 = "
scripts/data/prewarm_voronoi_mesh.py-40-                         "labelled synthetic scaling mesh (minutes, not "
scripts/data/prewarm_voronoi_mesh.py-41-                         "hours).")
scripts/data/prewarm_voronoi_mesh.py-42-    args = ap.parse_args()
--
scripts/data/prewarm_voronoi_mesh.py-50-    from legoesm.grids.voronoi import create_voronoi_mesh, prewarm_voronoi_cache
scripts/data/prewarm_voronoi_mesh.py-51-
scripts/data/prewarm_voronoi_mesh.py-52-    t0 = time.time()
scripts/data/prewarm_voronoi_mesh.py-53-    if args.level > 8:
scripts/data/prewarm_voronoi_mesh.py:54:        mesh = create_voronoi_mesh(args.level, lloyd_iterations=args.lloyd)
scripts/data/prewarm_voronoi_mesh.py-55-        path = "(big-mesh policy cache)"
scripts/data/prewarm_voronoi_mesh.py-56-    else:
scripts/data/prewarm_voronoi_mesh.py-57-        path = prewarm_voronoi_cache(args.level,
scripts/data/prewarm_voronoi_mesh.py:58:                                     lloyd_iterations=args.lloyd)
scripts/data/prewarm_voronoi_mesh.py-59-        mesh = None
scripts/data/prewarm_voronoi_mesh.py-60-    dt = time.time() - t0
scripts/data/prewarm_voronoi_mesh.py-61-    n = mesh.nCells if mesh is not None else 10 * 4 ** args.level + 2
scripts/data/prewarm_voronoi_mesh.py:62:    print(f"prewarmed level={args.level} lloyd={args.lloyd}: {n} cells "
scripts/data/prewarm_voronoi_mesh.py-63-          f"in {dt/60:.1f} min -> {path}")
scripts/data/prewarm_voronoi_mesh.py-64-    return 0
scripts/data/prewarm_voronoi_mesh.py-65-
scripts/data/prewarm_voronoi_mesh.py-66-
--
tests/parallel/test_ppermute_halo_exchange.py-142-        data).compile().as_text()
tests/parallel/test_ppermute_halo_exchange.py-143-    assert "collective-permute" in hlo and "all-gather" not in hlo
tests/parallel/test_ppermute_halo_exchange.py-144-
tests/parallel/test_ppermute_halo_exchange.py-145-
tests/parallel/test_ppermute_halo_exchange.py:146:def test_ppermute_schedule_covers_24_adjacencies_once():
tests/parallel/test_ppermute_halo_exchange.py-147-    """Invariant the ppermute correctness relies on: the 4 rounds cover all 24
tests/parallel/test_ppermute_halo_exchange.py-148-    directed face adjacencies exactly once, and each round is a permutation
tests/parallel/test_ppermute_halo_exchange.py-149-    (every face appears once as src and once as dst)."""
tests/parallel/test_ppermute_halo_exchange.py-150-    perms = cx._PPERMUTE_PERMS
--
tests/parallel/test_tiled_mass_divergence.py-93-                worst = max(worst, float(np.max(np.abs(t - g))))
tests/parallel/test_tiled_mass_divergence.py-94-
tests/parallel/test_tiled_mass_divergence.py-95-    # FMA-robust relative tolerance: the deep-pad + per-tile reconstruction
tests/parallel/test_tiled_mass_divergence.py-96-    # reorders the global's contiguous PPM arithmetic -> O(1e-13) ULP drift,
tests/parallel/test_tiled_mass_divergence.py:97:    # not algorithmic.  Tight enough to catch a halo-depth / index / upwind /
tests/parallel/test_tiled_mass_divergence.py-98-    # stagger bug.
tests/parallel/test_tiled_mass_divergence.py-99-    scale = float(np.max(np.abs(dh_g))) + 1e-300
tests/parallel/test_tiled_mass_divergence.py-100-    rel = worst / scale
tests/parallel/test_tiled_mass_divergence.py-101-    assert rel < 1e-10, (
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-32-#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-33-#             margin below it is the measured contention.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-34-#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-35-#             per replica.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:36:# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-37-# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-38-# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-39-# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-40-# table sacct prints at the end.  Absolute ms/step is therefore only
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-67-      --job-name="arm_$1" \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-68-    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-69-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-70-      --multicontroller --n-devices 32 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:71:      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-72-      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-73-      --out "$OUTDIR/$1.jsonl"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-74-  s=$?
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-75-  echo "[$1] exit=$s epoch=$(date +%s.%N)"
--
scripts/run/run_rcemip_long.py-779-        compute_terrain_metric, create_height_coordinate,
scripts/run/run_rcemip_long.py-780-    )
scripts/run/run_rcemip_long.py-781-    from legoesm.grids.voronoi import create_voronoi_mesh
scripts/run/run_rcemip_long.py-782-
scripts/run/run_rcemip_long.py:783:    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
scripts/run/run_rcemip_long.py-784-    hc = create_height_coordinate(NLEV, H_TOP)
scripts/run/run_rcemip_long.py-785-    tm = compute_terrain_metric(jnp.zeros(mesh.nCells), hc)
scripts/run/run_rcemip_long.py-786-    cfg = MPASCompressibleEulerConfig(
scripts/run/run_rcemip_long.py-787-        nu_del2=1.0e4, n_acoustic_substeps=4, fix_mass=False,
--
scripts/cluster/scaling_levante/prewarm_s10.sbatch-6-#SBATCH --cpus-per-task=8
scripts/cluster/scaling_levante/prewarm_s10.sbatch-7-#SBATCH --mem=120G
scripts/cluster/scaling_levante/prewarm_s10.sbatch-8-#SBATCH --time=10:00:00
scripts/cluster/scaling_levante/prewarm_s10.sbatch-9-#SBATCH --output=prewarm_s10.%j.log
scripts/cluster/scaling_levante/prewarm_s10.sbatch:10:# Prewarm subdiv-10 lloyd=0 (10.5M cells, ~7 GB npz) into the shared
scripts/cluster/scaling_levante/prewarm_s10.sbatch-11-# mesh cache — unlocks MPAS at 128-224 GPUs ABOVE the ~30k tile floor
scripts/cluster/scaling_levante/prewarm_s10.sbatch-12-# (np128 = 81.9k, np224 = 46.8k cells/GPU).  s9 (2.62M) built in 67 min;
scripts/cluster/scaling_levante/prewarm_s10.sbatch-13-# s10 estimated ~4-5 h.
scripts/cluster/scaling_levante/prewarm_s10.sbatch-14-set -uo pipefail
--
scripts/cluster/scaling_levante/prewarm_s10.sbatch-18-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/prewarm_s10.sbatch-19-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/prewarm_s10.sbatch-20-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/prewarm_s10.sbatch-21-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
scripts/cluster/scaling_levante/prewarm_s10.sbatch:22:"$PY" scripts/data/prewarm_voronoi_mesh.py --level 10 --lloyd 0
--
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-8-#SBATCH --exclusive
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-10-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-11-#SBATCH --output=mpas_s10_192.%j.log
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:12:# HUNDREDS-OF-GPUS MPAS, rung 1: subdiv-10 (10.49M cells, lloyd=0
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-13-# synthetic, prewarmed job 26628074) at 128 GPUs = 81.9k cells/GPU —
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-14-# the SAME near-matched tile as s8@np8 (6.58 ms) and s9@np32 (12.47 ms),
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-15-# extending the 4x-devices weak series a third rung.
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-16-# Falsifiability, written BEFORE submit:
--
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-32-JAX_ENABLE_X64=0 srun --ntasks=192 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-33-    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-34-  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-35-    --multicontroller --n-devices 192 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:36:    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-37-    --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-38-    --out "$OUTDIR/np192.jsonl" || { echo "np192 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-39-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-40-import json,math,sys
--
tests/unit/test_voronoi_big_mesh_policy.py-27-
tests/unit/test_voronoi_big_mesh_policy.py-28-def test_hard_cap_refused_regardless_of_optin(big_at_2, monkeypatch):
tests/unit/test_voronoi_big_mesh_policy.py-29-    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
tests/unit/test_voronoi_big_mesh_policy.py-30-    with pytest.raises(ValueError, match="unsupported"):
tests/unit/test_voronoi_big_mesh_policy.py:31:        V.create_voronoi_mesh(5, lloyd_iterations=0)
tests/unit/test_voronoi_big_mesh_policy.py-32-
tests/unit/test_voronoi_big_mesh_policy.py-33-
tests/unit/test_voronoi_big_mesh_policy.py-34-def test_miss_rejected_without_optin(big_at_2):
tests/unit/test_voronoi_big_mesh_policy.py-35-    with pytest.raises(ValueError, match="prewarm"):
tests/unit/test_voronoi_big_mesh_policy.py:36:        V.create_voronoi_mesh(3, lloyd_iterations=0)
tests/unit/test_voronoi_big_mesh_policy.py-37-
tests/unit/test_voronoi_big_mesh_policy.py-38-
tests/unit/test_voronoi_big_mesh_policy.py-39-def test_optin_builds_then_hit_bypasses_policy(big_at_2, monkeypatch):
tests/unit/test_voronoi_big_mesh_policy.py-40-    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
tests/unit/test_voronoi_big_mesh_policy.py:41:    m1 = V.create_voronoi_mesh(3, lloyd_iterations=0)
tests/unit/test_voronoi_big_mesh_policy.py-42-    # hit path: opt-in removed, still loads (cache hit is always admissible)
tests/unit/test_voronoi_big_mesh_policy.py-43-    monkeypatch.delenv(V._BIG_MESH_BUILD_ENV)
tests/unit/test_voronoi_big_mesh_policy.py:44:    m2 = V.create_voronoi_mesh(3, lloyd_iterations=0)
tests/unit/test_voronoi_big_mesh_policy.py-45-    assert m1.nCells == m2.nCells
tests/unit/test_voronoi_big_mesh_policy.py-46-
tests/unit/test_voronoi_big_mesh_policy.py-47-
tests/unit/test_voronoi_big_mesh_policy.py-48-def test_disabled_cache_refused_for_big(big_at_2, monkeypatch):
tests/unit/test_voronoi_big_mesh_policy.py-49-    monkeypatch.setenv(V._MESH_CACHE_DISABLE_ENV, "1")
tests/unit/test_voronoi_big_mesh_policy.py-50-    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
tests/unit/test_voronoi_big_mesh_policy.py-51-    with pytest.raises(ValueError, match="cache"):
tests/unit/test_voronoi_big_mesh_policy.py:52:        V.create_voronoi_mesh(3, lloyd_iterations=0)
tests/unit/test_voronoi_big_mesh_policy.py-53-
tests/unit/test_voronoi_big_mesh_policy.py-54-
tests/unit/test_voronoi_big_mesh_policy.py-55-def test_single_builder_concurrency(big_at_2, monkeypatch):
tests/unit/test_voronoi_big_mesh_policy.py-56-    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
tests/unit/test_voronoi_big_mesh_policy.py-57-    results, errors = [], []
tests/unit/test_voronoi_big_mesh_policy.py-58-
tests/unit/test_voronoi_big_mesh_policy.py-59-    def worker():
tests/unit/test_voronoi_big_mesh_policy.py-60-        try:
tests/unit/test_voronoi_big_mesh_policy.py:61:            results.append(V.create_voronoi_mesh(3, lloyd_iterations=0))
tests/unit/test_voronoi_big_mesh_policy.py-62-        except Exception as e:  # noqa: BLE001
tests/unit/test_voronoi_big_mesh_policy.py-63-            errors.append(e)
tests/unit/test_voronoi_big_mesh_policy.py-64-
tests/unit/test_voronoi_big_mesh_policy.py-65-    threads = [threading.Thread(target=worker) for _ in range(3)]
--
tests/unit/test_voronoi_big_mesh_policy.py-82-    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
tests/unit/test_voronoi_big_mesh_policy.py-83-    with open(f"{cache_path}.lock", "w") as f:
tests/unit/test_voronoi_big_mesh_policy.py-84-        f.write("999999")
tests/unit/test_voronoi_big_mesh_policy.py-85-    with pytest.warns(UserWarning, match="stale"):
tests/unit/test_voronoi_big_mesh_policy.py:86:        m = V.create_voronoi_mesh(3, lloyd_iterations=0)
tests/unit/test_voronoi_big_mesh_policy.py-87-    assert m.nCells == 10 * 4 ** 3 + 2
--
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-27-JAX_ENABLE_X64=0 srun --ntasks=192 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-28-    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-29-  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-30-    --multicontroller --n-devices 192 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:31:    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-32-    --partition-method sfc --reorder-for 192 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-33-    --out "$OUTDIR/np192.jsonl" || { echo "np192 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-34-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-35-import json,math,sys
--
tests/parallel/test_mpas_atm_native_step.py-78-    )
tests/parallel/test_mpas_atm_native_step.py-79-    from legoesm.grids.vertical import create_sigma_coordinate
tests/parallel/test_mpas_atm_native_step.py-80-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/parallel/test_mpas_atm_native_step.py-81-    from legoesm.parallel.voronoi_partition import (
tests/parallel/test_mpas_atm_native_step.py:82:        reorder_voronoi_for_sharding,
tests/parallel/test_mpas_atm_native_step.py-83-    )
tests/parallel/test_mpas_atm_native_step.py-84-
tests/parallel/test_mpas_atm_native_step.py-85-    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
tests/parallel/test_mpas_atm_native_step.py-86-
tests/parallel/test_mpas_atm_native_step.py-87-    mesh = create_voronoi_mesh(subdivision_level=_SUBDIV)
tests/parallel/test_mpas_atm_native_step.py:88:    mesh = reorder_voronoi_for_sharding(mesh, reorder_for)
tests/parallel/test_mpas_atm_native_step.py-89-    sigma = create_sigma_coordinate(_NLEV)
tests/parallel/test_mpas_atm_native_step.py-90-    cfg = MPASPrimitiveEquationConfig(
tests/parallel/test_mpas_atm_native_step.py-91-        nu_del4=1e16, nu_del4_ps=1e16,
tests/parallel/test_mpas_atm_native_step.py-92-        fix_mass=True, pv_scheme="energy",
--
tests/parallel/test_mpas_atm_native_step.py-542-        from jax.sharding import PartitionSpec as P
tests/parallel/test_mpas_atm_native_step.py-543-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/parallel/test_mpas_atm_native_step.py-544-        from legoesm.parallel.shard_map_compat import shard_map
tests/parallel/test_mpas_atm_native_step.py-545-        from legoesm.parallel.sharded_dynamics import (
tests/parallel/test_mpas_atm_native_step.py:546:            _build_ppermute_schedule,
tests/parallel/test_mpas_atm_native_step.py-547-            _build_voronoi_partition_infra,
tests/parallel/test_mpas_atm_native_step.py-548-            _pack_cell_state,
tests/parallel/test_mpas_atm_native_step.py-549-            _ppermute_halo_fill,
tests/parallel/test_mpas_atm_native_step.py-550-            _unpack_cell_state,
tests/parallel/test_mpas_atm_native_step.py-551-        )
tests/parallel/test_mpas_atm_native_step.py-552-        from legoesm.parallel.voronoi_partition import (
tests/parallel/test_mpas_atm_native_step.py:553:            reorder_voronoi_for_sharding,
tests/parallel/test_mpas_atm_native_step.py-554-        )
tests/parallel/test_mpas_atm_native_step.py-555-
tests/parallel/test_mpas_atm_native_step.py-556-        nlev = 3
tests/parallel/test_mpas_atm_native_step.py-557-        mesh = create_voronoi_mesh(subdivision_level=2)
tests/parallel/test_mpas_atm_native_step.py:558:        mesh = reorder_voronoi_for_sharding(mesh, n_dev)
tests/parallel/test_mpas_atm_native_step.py-559-        nCells, nEdges = mesh.nCells, mesh.nEdges
tests/parallel/test_mpas_atm_native_step.py-560-        cells_per, edges_per = nCells // n_dev, nEdges // n_dev
tests/parallel/test_mpas_atm_native_step.py-561-
tests/parallel/test_mpas_atm_native_step.py-562-        (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
tests/parallel/test_mpas_atm_native_step.py-563-         cell_owner) = _build_voronoi_partition_infra(
tests/parallel/test_mpas_atm_native_step.py-564-            mesh, n_dev, halo_depth=3)
tests/parallel/test_mpas_atm_native_step.py:565:        sched = _build_ppermute_schedule(
tests/parallel/test_mpas_atm_native_step.py-566-            partitions, cell_owner, n_dev, cells_per, edges_per,
tests/parallel/test_mpas_atm_native_step.py-567-            max_lc, max_le)
tests/parallel/test_mpas_atm_native_step.py-568-        assert sched["n_rounds"] >= 1, "no comm rounds — test is vacuous"
tests/parallel/test_mpas_atm_native_step.py-569-        if n_dev >= 3:
--
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-8-#SBATCH --mem=120G
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-9-#SBATCH --time=01:00:00
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-10-#SBATCH --output=mpas_bound.%j.log
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-11-# MPAS same-tile nd=1 compute anchor for the panel's calibrated bound:
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:12:# subdiv-6 lloyd0 GLOBAL mesh (40,962 natural cells) ~= the 41k
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:13:# cells/GPU tile of the measured rows s8-lloyd0@np16 (6.43 ms, job
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-14-# 26628076) and s9@np64 (9.60 ms, job 26600095).  A single-device run
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-15-# has zero inter-device traffic, so its fused step time IS the compute
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:16:# term at this tile (mesh-family lloyd0 matched; padding differs by
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-17-# construction — model-grade, recorded).
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-18-set -uo pipefail
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-19-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-20-export JAX_PLATFORMS=cuda,cpu
--
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-28-rc=0
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-29-JAX_ENABLE_X64=0 srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-30-  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-31-    --n-devices 1 --subdivision 6 --nlev 26 --steps 12 --warmup 3 \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:32:    --lloyd 0 --partition-method sfc \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-33-    --out "$OUTDIR/s6_nd1.jsonl" || { echo "s6_nd1 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-34-"$PY" -c "
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-35-import json,math,sys
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-36-try:
--
scripts/run/run_w2_mpas_convergence.py-32-    compute_error_norms_mpas,
scripts/run/run_w2_mpas_convergence.py-33-)
scripts/run/run_w2_mpas_convergence.py-34-
scripts/run/run_w2_mpas_convergence.py-35-
scripts/run/run_w2_mpas_convergence.py:36:def run_tc2(level: int, duration_days: float = 1.0, lloyd_iters: int = 50):
scripts/run/run_w2_mpas_convergence.py-37-    """Run TC2 at a given mesh level and return error norms."""
scripts/run/run_w2_mpas_convergence.py-38-    print(f"\n{'='*60}")
scripts/run/run_w2_mpas_convergence.py:39:    print(f"Level {level}: generating mesh (lloyd_iterations={lloyd_iters}) ...")
scripts/run/run_w2_mpas_convergence.py-40-    t0 = time.time()
scripts/run/run_w2_mpas_convergence.py:41:    mesh = create_voronoi_mesh(level, lloyd_iterations=lloyd_iters)
scripts/run/run_w2_mpas_convergence.py-42-    print(f"  nCells={mesh.nCells}, nEdges={mesh.nEdges}, nVertices={mesh.nVertices}")
scripts/run/run_w2_mpas_convergence.py-43-    print(f"  Mesh generation: {time.time()-t0:.1f}s")
scripts/run/run_w2_mpas_convergence.py-44-
scripts/run/run_w2_mpas_convergence.py-45-    # Compute characteristic cell spacing
--
tests/unit/test_mpas_nh_radiation_microphysics.py-41-@pytest.fixture(scope="module")
tests/unit/test_mpas_nh_radiation_microphysics.py-42-def mpas_setup():
tests/unit/test_mpas_nh_radiation_microphysics.py-43-    """Small MPAS NH state with 3 moist tracers seeded at the surface."""
tests/unit/test_mpas_nh_radiation_microphysics.py-44-    nlev = 8
tests/unit/test_mpas_nh_radiation_microphysics.py:45:    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
tests/unit/test_mpas_nh_radiation_microphysics.py-46-    hc = create_height_coordinate(nlev, H=20_000.0)
tests/unit/test_mpas_nh_radiation_microphysics.py-47-    tm = compute_terrain_metric(jnp.zeros(mesh.nCells), hc)
tests/unit/test_mpas_nh_radiation_microphysics.py-48-    tracers = jnp.zeros((mesh.nCells, nlev, 3), dtype=jnp.float64)
tests/unit/test_mpas_nh_radiation_microphysics.py-49-    tracers = tracers.at[..., -1, 0].set(0.01)
--
scripts/run/mpas_4dvar_single/common.py-186-def make_grid_sigma_model(model_config: dict):
scripts/run/mpas_4dvar_single/common.py-187-    mesh = create_grid(
scripts/run/mpas_4dvar_single/common.py-188-        "mpas",
scripts/run/mpas_4dvar_single/common.py-189-        int(model_config["resolution"]),
scripts/run/mpas_4dvar_single/common.py:190:        lloyd_iterations=int(model_config["lloyd_iterations"]),
scripts/run/mpas_4dvar_single/common.py-191-    )
scripts/run/mpas_4dvar_single/common.py-192-    sigma = create_sigma_coordinate(int(model_config["levels"]))
scripts/run/mpas_4dvar_single/common.py-193-    dycore = DycoreConfig(
scripts/run/mpas_4dvar_single/common.py-194-        model_type="hydrostatic",
--
tests/test_dispatch_hardening.py-208-        ("packages/core/legoesm/parallel/runtime.py", "halo_exchange"),
tests/test_dispatch_hardening.py-209-        ("packages/core/legoesm/parallel/voronoi_mpi.py", "gather_voronoi_field"),
tests/test_dispatch_hardening.py-210-        ("packages/core/legoesm/parallel/voronoi_mpi.py", "initialize_voronoi_mpi"),
tests/test_dispatch_hardening.py-211-        ("packages/core/legoesm/parallel/voronoi_partition.py", "partition_voronoi_mesh"),
tests/test_dispatch_hardening.py:212:        ("packages/core/legoesm/parallel/voronoi_partition.py", "reorder_voronoi_for_sharding"),
tests/test_dispatch_hardening.py-213-        ("packages/core/legoesm/timestepping/dispatch.py", "dispatch_integrator"),
tests/test_dispatch_hardening.py-214-        ("packages/core/legoesm/timestepping/split_explicit.py", "split_explicit_step"),
tests/test_dispatch_hardening.py-215-        ("packages/coupler/legoesm/coupler/coupler.py", "ocean_tile_response"),
tests/test_dispatch_hardening.py-216-        ("packages/coupler/legoesm/coupler/lake/two_layer_lake.py", "step_lake"),
--
tests/unit/test_mpas_vertical_checkerboard.py-47-NLEV = 24
tests/unit/test_mpas_vertical_checkerboard.py-48-
tests/unit/test_mpas_vertical_checkerboard.py-49-
tests/unit/test_mpas_vertical_checkerboard.py-50-def _mesh():
tests/unit/test_mpas_vertical_checkerboard.py:51:    return create_voronoi_mesh(2, lloyd_iterations=5)
tests/unit/test_mpas_vertical_checkerboard.py-52-
tests/unit/test_mpas_vertical_checkerboard.py-53-
tests/unit/test_mpas_vertical_checkerboard.py-54-def _resting_state(mesh, nlev, seed_2dz=0.0, T0=250.0):
tests/unit/test_mpas_vertical_checkerboard.py-55-    """Resting (u=0) isothermal column + optional 2Δσ T perturbation."""
--
tests/ocean/unit/test_omip2_applicator.py-290-    from legoesm.ocean.coupler import compute_omip2_surface_forcing
tests/ocean/unit/test_omip2_applicator.py-291-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/ocean/unit/test_omip2_applicator.py-292-    from legoesm.ocean.vertical import create_ocean_z_star
tests/ocean/unit/test_omip2_applicator.py-293-    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
tests/ocean/unit/test_omip2_applicator.py:294:    mesh = create_voronoi_mesh(2, lloyd_iterations=2)   # 162 cells, tiny
tests/ocean/unit/test_omip2_applicator.py-295-    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
tests/ocean/unit/test_omip2_applicator.py-296-    state = rest_state_mpas_ocean(mesh, z, H_max=4000.0)
tests/ocean/unit/test_omip2_applicator.py-297-    forcing = _uniform_wind_forcing(u_east=8.0)
tests/ocean/unit/test_omip2_applicator.py-298-    sf = compute_omip2_surface_forcing(
--
tests/ocean/unit/test_advection_grad_underflow.py-172-            compute_upup_cells,
tests/ocean/unit/test_advection_grad_underflow.py-173-            tvd_tracer_to_edges,
tests/ocean/unit/test_advection_grad_underflow.py-174-        )
tests/ocean/unit/test_advection_grad_underflow.py-175-
tests/ocean/unit/test_advection_grad_underflow.py:176:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/ocean/unit/test_advection_grad_underflow.py-177-        upup_pos, upup_neg = compute_upup_cells(mesh)
tests/ocean/unit/test_advection_grad_underflow.py-178-        nlev = 3
tests/ocean/unit/test_advection_grad_underflow.py-179-        key = jax.random.PRNGKey(11)
tests/ocean/unit/test_advection_grad_underflow.py-180-        # near-uniform tracer (the NaN regime) + an exactly-uniform level
--
tests/ocean/unit/test_ocean_differentiability.py-27-    @pytest.fixture(scope="class")
tests/ocean/unit/test_ocean_differentiability.py-28-    def mpas_setup(self):
tests/ocean/unit/test_ocean_differentiability.py-29-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/ocean/unit/test_ocean_differentiability.py-30-        from legoesm.core.operators_voronoi import smagorinsky_biharmonic_3d
tests/ocean/unit/test_ocean_differentiability.py:31:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=10)
tests/ocean/unit/test_ocean_differentiability.py-32-        nlev = 3
tests/ocean/unit/test_ocean_differentiability.py-33-        nEdges = mesh.nEdges
tests/ocean/unit/test_ocean_differentiability.py-34-        key = jax.random.PRNGKey(42)
tests/ocean/unit/test_ocean_differentiability.py-35-        u_edge = 0.01 * jax.random.normal(key, (nEdges, nlev), dtype=jnp.float64)
--
tests/ocean/unit/test_ocean_differentiability.py-511-    try:
tests/ocean/unit/test_ocean_differentiability.py-512-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/ocean/unit/test_ocean_differentiability.py-513-        from legoesm.core.operators_voronoi import smagorinsky_biharmonic_3d
tests/ocean/unit/test_ocean_differentiability.py-514-
tests/ocean/unit/test_ocean_differentiability.py:515:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=10)
tests/ocean/unit/test_ocean_differentiability.py-516-        key = jax.random.PRNGKey(42)
tests/ocean/unit/test_ocean_differentiability.py-517-        u_edge = 0.01 * jax.random.normal(
tests/ocean/unit/test_ocean_differentiability.py-518-            key, (mesh.nEdges, 3), dtype=jnp.float64,
tests/ocean/unit/test_ocean_differentiability.py-519-        )
--
scripts/cluster/scaling_derecho/README.md-508-
scripts/cluster/scaling_derecho/README.md-509-`gpu_multinode_scaling.pbs` gained lane E (`RUN_MPAS=1`, default on): the
scripts/cluster/scaling_derecho/README.md-510-icosahedral MPAS PE dycore over `jax.distributed` + NCCL via
scripts/cluster/scaling_derecho/README.md-511-`scripts/bench/bench_mpas_spmd_scaling.py` — cell-partition reorder
scripts/cluster/scaling_derecho/README.md:512:(`reorder_voronoi_for_sharding`, Hilbert-SFC pinned for cross-process
scripts/cluster/scaling_derecho/README.md-513-determinism) + `make_voronoi_sharded_step` ppermute halos. 6 processes
scripts/cluster/scaling_derecho/README.md-514-(2 nodes x 3 GPUs): `nCells = 10*4^L + 2` admits 1/2/3/6 even splits at
scripts/cluster/scaling_derecho/README.md-515-every level. A subdiv-4 smoke with `--parity-gate --check-conservation`
scripts/cluster/scaling_derecho/README.md-516-runs before the timed `ICO_LEVEL` (default L7 = 163842 cells, ~27k
--
tests/unit/test_mpas_variable_resolution.py-15-
tests/unit/test_mpas_variable_resolution.py-16-
tests/unit/test_mpas_variable_resolution.py-17-def test_density_none_reproduces_uniform_mesh() -> None:
tests/unit/test_mpas_variable_resolution.py-18-    """density_fn=None is byte-identical to the legacy uniform SCVT."""
tests/unit/test_mpas_variable_resolution.py:19:    a = create_voronoi_mesh(3, lloyd_iterations=8)
tests/unit/test_mpas_variable_resolution.py:20:    b = create_voronoi_mesh(3, lloyd_iterations=8, density_fn=None)
tests/unit/test_mpas_variable_resolution.py-21-    assert np.array_equal(np.asarray(a.latCell), np.asarray(b.latCell))
tests/unit/test_mpas_variable_resolution.py-22-    assert np.array_equal(np.asarray(a.areaCell), np.asarray(b.areaCell))
tests/unit/test_mpas_variable_resolution.py-23-
tests/unit/test_mpas_variable_resolution.py-24-
--
tests/unit/test_mpas_variable_resolution.py-29-    def density(lat, lon):
tests/unit/test_mpas_variable_resolution.py-30-        d = _greatcircle(np.asarray(lat), np.asarray(lon), lat0, lon0)
tests/unit/test_mpas_variable_resolution.py-31-        return 1.0 + 9.0 * np.exp(-(d ** 2) / (0.35 ** 2))  # 10x finer at centre
tests/unit/test_mpas_variable_resolution.py-32-
tests/unit/test_mpas_variable_resolution.py:33:    mesh = create_voronoi_mesh(4, lloyd_iterations=60, density_fn=density)
tests/unit/test_mpas_variable_resolution.py-34-    lat_c = np.asarray(mesh.latCell)
tests/unit/test_mpas_variable_resolution.py-35-    lon_c = np.asarray(mesh.lonCell)
tests/unit/test_mpas_variable_resolution.py-36-    area = np.asarray(mesh.areaCell)
tests/unit/test_mpas_variable_resolution.py-37-
--
tests/unit/test_mpas_variable_resolution.py-42-    # (1) refinement DIRECTION: the high-density region has smaller cells
tests/unit/test_mpas_variable_resolution.py-43-    assert area[near].mean() < area[far].mean()
tests/unit/test_mpas_variable_resolution.py-44-    # (2) genuine VARIABLE resolution: cells span a real size range (not uniform).
tests/unit/test_mpas_variable_resolution.py-45-    # Density-weighted Lloyd converges linearly, so the achieved contrast grows
tests/unit/test_mpas_variable_resolution.py:46:    # with lloyd_iterations; for strong/precise refinement use a JIGSAW-built mesh
tests/unit/test_mpas_variable_resolution.py-47-    # via load_mpas_mesh.  At 60 iters the smallest cell is well under the largest.
tests/unit/test_mpas_variable_resolution.py-48-    assert area.max() / area.min() > 1.5
tests/unit/test_mpas_variable_resolution.py-49-
tests/unit/test_mpas_variable_resolution.py-50-
tests/unit/test_mpas_variable_resolution.py-51-def test_density_must_be_positive_and_finite() -> None:
tests/unit/test_mpas_variable_resolution.py-52-    with pytest.raises(ValueError, match="positive"):
tests/unit/test_mpas_variable_resolution.py:53:        create_voronoi_mesh(2, lloyd_iterations=5,
tests/unit/test_mpas_variable_resolution.py-54-                            density_fn=lambda lat, lon: -1.0)
tests/unit/test_mpas_variable_resolution.py-55-
tests/unit/test_mpas_variable_resolution.py-56-
tests/unit/test_mpas_variable_resolution.py:57:def test_density_requires_lloyd_relaxation() -> None:
tests/unit/test_mpas_variable_resolution.py-58-    with pytest.raises(ValueError, match="density_fn requires"):
tests/unit/test_mpas_variable_resolution.py:59:        create_voronoi_mesh(2, lloyd_iterations=0,
tests/unit/test_mpas_variable_resolution.py-60-                            density_fn=lambda lat, lon: 1.0)
tests/unit/test_mpas_variable_resolution.py-61-
tests/unit/test_mpas_variable_resolution.py-62-
tests/unit/test_mpas_variable_resolution.py-63-def test_variable_resolution_via_create_grid_factory() -> None:
tests/unit/test_mpas_variable_resolution.py-64-    """The uniform create_grid factory forwards density_fn -> MPAS variable-res."""
tests/unit/test_mpas_variable_resolution.py-65-    from legoesm.grids.factory import create_grid
tests/unit/test_mpas_variable_resolution.py-66-
tests/unit/test_mpas_variable_resolution.py:67:    mesh = create_grid("mpas", 3, lloyd_iterations=20,
tests/unit/test_mpas_variable_resolution.py-68-                       density_fn=lambda lat, lon: 1.0 + 5.0 * np.exp(-(lat ** 2) / 0.2))
tests/unit/test_mpas_variable_resolution.py-69-    # cells cluster near the equator (lat=0) -> smaller there than near the poles
tests/unit/test_mpas_variable_resolution.py-70-    lat_c = np.asarray(mesh.latCell)
tests/unit/test_mpas_variable_resolution.py-71-    area = np.asarray(mesh.areaCell)
--
tests/unit/test_conservative_regrid_unstructured.py-28-    def setUpClass(cls):
tests/unit/test_conservative_regrid_unstructured.py-29-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_conservative_regrid_unstructured.py-30-        # Two global SCVT meshes at different resolution (mesh quality is
tests/unit/test_conservative_regrid_unstructured.py-31-        # irrelevant to conservation, so few Lloyd iterations keep it fast).
tests/unit/test_conservative_regrid_unstructured.py:32:        cls.coarse = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)  # 162
tests/unit/test_conservative_regrid_unstructured.py:33:        cls.fine = create_voronoi_mesh(subdivision_level=3, lloyd_iterations=5)    # 642
tests/unit/test_conservative_regrid_unstructured.py-34-
tests/unit/test_conservative_regrid_unstructured.py-35-    def _weights(self, src, dst, n_sub=6):
tests/unit/test_conservative_regrid_unstructured.py-36-        from legoesm.grids.conservative_regrid_unstructured import (
tests/unit/test_conservative_regrid_unstructured.py-37-            compute_mpas_to_mpas_weights,
--
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch-33-  --adaptive-implicit-vertadv --momentum-rk3 --geothermal \
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch-34-  --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch-35-  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch-36-  --years 0.3 --snapshot-every-days 30 --diag-every-days 5"
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch:37:GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch-38-LEVERS="--tracer-advection superbee --runoff-depth-spread-m 150"   # nogate (no --river-mouth-restoring-gate)
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch-39-KPP="--kpp-ri-crit 0.5 --kpp-cv 2.5"
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch-40-
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch-41-LABEL=ico7_kppdeep_ri05cv25_r2
--
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch-36-SSSCLIM=$INP/sss_climatology_for_restoring.nc
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch-37-ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch-38-echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch-39-echo "=== worktree (fix tree) ==="; git -C $WT log --oneline -1
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch:40:GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch-41-COMMON="--nemo-vertical --partial-cell --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.3 --snapshot-every-days 30 --diag-every-days 5"
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch-42-LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch-43-PARITY="--bottom-drag-scheme nemo_quadratic"
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch-44-IC="--woa-init --woa-t $WTIC --woa-s $WSIC --sss-restore-file $SSSCLIM --ice-init $ICEIC"
--
tests/unit/test_grid_dycore_fixes.py-88-
tests/unit/test_grid_dycore_fixes.py-89-    @pytest.fixture
tests/unit/test_grid_dycore_fixes.py-90-    def mesh(self):
tests/unit/test_grid_dycore_fixes.py-91-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_grid_dycore_fixes.py:92:        return create_voronoi_mesh(1, lloyd_iterations=10)
tests/unit/test_grid_dycore_fixes.py-93-
tests/unit/test_grid_dycore_fixes.py-94-    def test_weight_antisymmetry(self, mesh):
tests/unit/test_grid_dycore_fixes.py-95-        """Dimensionless Thuburn weight antisymmetry (Ringler 2010 Eq. 49).
tests/unit/test_grid_dycore_fixes.py-96-
--
tests/unit/test_cdgrid_fv3_regression.py-1346-                            f"class names instead; readers can grep."
tests/unit/test_cdgrid_fv3_regression.py-1347-                        )
tests/unit/test_cdgrid_fv3_regression.py-1348-
tests/unit/test_cdgrid_fv3_regression.py-1349-    def test_d2a2c_vect_non_duogrid_cube_vertex_gap_architectural_bound(self):
tests/unit/test_cdgrid_fv3_regression.py:1350:        """Iter-128 (Priority 3): guard the halo-depth invariant that
tests/unit/test_cdgrid_fv3_regression.py-1351-        locks out the deepest Fortran cube-vertex override.
tests/unit/test_cdgrid_fv3_regression.py-1352-
tests/unit/test_cdgrid_fv3_regression.py-1353-        Fortran sw_core.F90:3527-3545 writes `utmp(-2..0, 0)` at three
tests/unit/test_cdgrid_fv3_regression.py-1354-        halo cells (depths 1, 2, 3 west of interior).  Porting the
--
tests/unit/test_component_factory.py-393-            ("latlon", 16, "shallow_water", "latlon_cgrid", {},
tests/unit/test_component_factory.py-394-             "CGridLatLonShallowWaterModel", "grid"),
tests/unit/test_component_factory.py-395-            # mpas exposes no shallow_water in the driver matrix -> hydrostatic;
tests/unit/test_component_factory.py-396-            # the TRiSK mesh is held on the model as `.mesh`, not `.grid`.
tests/unit/test_component_factory.py:397:            ("mpas", 1, "hydrostatic", "mpas", {"lloyd_iterations": 2},
tests/unit/test_component_factory.py-398-             "MPASPrimitiveEquationModel", "mesh"),
tests/unit/test_component_factory.py-399-            # Non-hydrostatic branches take (grid, height_coord, terrain_metric),
tests/unit/test_component_factory.py-400-            # not sigma_coord — these two rows caught the factory passing
tests/unit/test_component_factory.py-401-            # sigma_coord= to constructors that do not accept it.
--
tests/unit/test_component_factory.py-406-                    not jax.config.read("jax_enable_x64"),
tests/unit/test_component_factory.py-407-                    reason="spectral/Gaussian needs JAX_ENABLE_X64=1",
tests/unit/test_component_factory.py-408-                ),
tests/unit/test_component_factory.py-409-            ),
tests/unit/test_component_factory.py:410:            ("mpas", 1, "nonhydrostatic", "mpas", {"lloyd_iterations": 2},
tests/unit/test_component_factory.py-411-             "MPASCompressibleEulerModel", "mesh"),
tests/unit/test_component_factory.py-412-        ],
tests/unit/test_component_factory.py-413-    )
tests/unit/test_component_factory.py-414-    def test_dycore_builds_on_factory_grid(
--
tests/unit/test_component_factory.py-858-        from legoesm.grids.factory import create_grid
tests/unit/test_component_factory.py-859-        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
tests/unit/test_component_factory.py-860-            MPASPrimitiveEquationConfig,
tests/unit/test_component_factory.py-861-        )
tests/unit/test_component_factory.py:862:        grid = create_grid("mpas", 1, lloyd_iterations=2)
tests/unit/test_component_factory.py-863-        config = _make_config(
tests/unit/test_component_factory.py-864-            model_type="hydrostatic", discretization="mpas",
tests/unit/test_component_factory.py-865-            grid_type="mpas", nlev=2,
tests/unit/test_component_factory.py-866-        )
--
tests/unit/test_component_factory.py-967-
tests/unit/test_component_factory.py-968-    @pytest.mark.parametrize("flag", [True, False])
tests/unit/test_component_factory.py-969-    def test_mpas_nh_forwards_fix_mass(self, flag):
tests/unit/test_component_factory.py-970-        from legoesm.grids.factory import create_grid
tests/unit/test_component_factory.py:971:        grid = create_grid("mpas", 1, lloyd_iterations=2)
tests/unit/test_component_factory.py-972-        config = ExperimentConfig(
tests/unit/test_component_factory.py-973-            grid=GridConfig(grid_type="mpas", resolution=1, nlev=2),
tests/unit/test_component_factory.py-974-            dycore=DycoreConfig(
tests/unit/test_component_factory.py-975-                model_type="nonhydrostatic", discretization="mpas",
--
tests/unit/test_component_factory.py-1033-        assert model.config.anchor_mass_to_initial is False
tests/unit/test_component_factory.py-1034-
tests/unit/test_component_factory.py-1035-    def test_mpas_nh_conservation_fixer_false_overrides(self):
tests/unit/test_component_factory.py-1036-        from legoesm.grids.factory import create_grid
tests/unit/test_component_factory.py:1037:        grid = create_grid("mpas", 1, lloyd_iterations=2)
tests/unit/test_component_factory.py-1038-        config = ExperimentConfig(
tests/unit/test_component_factory.py-1039-            grid=GridConfig(grid_type="mpas", resolution=1, nlev=2),
tests/unit/test_component_factory.py-1040-            dycore=DycoreConfig(
tests/unit/test_component_factory.py-1041-                model_type="nonhydrostatic", discretization="mpas",
--
tests/unit/test_component_factory.py-1049-    # codex round 2: the MPAS hydrostatic PE and plane NH branches
tests/unit/test_component_factory.py-1050-    # pre-dated the audit but had the same ungated forwarding.
tests/unit/test_component_factory.py-1051-    def test_mpas_pe_conservation_fixer_false_overrides(self):
tests/unit/test_component_factory.py-1052-        from legoesm.grids.factory import create_grid
tests/unit/test_component_factory.py:1053:        grid = create_grid("mpas", 1, lloyd_iterations=2)
tests/unit/test_component_factory.py-1054-        config = ExperimentConfig(
tests/unit/test_component_factory.py-1055-            grid=GridConfig(grid_type="mpas", resolution=1, nlev=2),
tests/unit/test_component_factory.py-1056-            dycore=DycoreConfig(
tests/unit/test_component_factory.py-1057-                model_type="hydrostatic", discretization="mpas",
--
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py-40-
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py-41-@pytest.fixture(scope="module")
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py-42-def mesh(_fp64_policy):
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py-43-    """Level-3 icosahedral mesh (642 cells)."""
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:44:    return create_voronoi_mesh(3, lloyd_iterations=50)
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py-45-
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py-46-
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py-47-class TestWilliamsonTC2:
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py-48-    """Williamson Test Case 2: steady-state geostrophic flow."""
--
tests/unit/test_dycore_component.py-227-    from legoesm.grids.vertical import create_sigma_coordinate
tests/unit/test_dycore_component.py-228-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_dycore_component.py-229-
tests/unit/test_dycore_component.py-230-    nlev = 5
tests/unit/test_dycore_component.py:231:    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
tests/unit/test_dycore_component.py-232-    sigma = create_sigma_coordinate(nlev)
tests/unit/test_dycore_component.py-233-    model = MPASPrimitiveEquationModel(mesh, sigma)
tests/unit/test_dycore_component.py-234-    t_data = 250.0 * jnp.ones((mesh.nCells, nlev)) + jax.random.normal(
tests/unit/test_dycore_component.py-235-        jax.random.PRNGKey(11), (mesh.nCells, nlev))
--
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch-21-WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch-22-WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch-23-echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch-24-git -C $WT log --oneline -1
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch:25:GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch-26-COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv \
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch-27-  --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch-28-  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch-29-  --prognostic-ice-dynamics free_drift --years 0.3 --snapshot-every-days 15 --diag-every-days 5"
--
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch-33-SSSCLIM=$INP/sss_climatology_for_restoring.nc
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch-34-ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch-35-echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch-36-echo "=== repo ==="; git -C $ROOT log --oneline -1
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch:37:GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch-38-COMMON="--nemo-vertical --partial-cell --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.085 --snapshot-every-days 30 --diag-every-days 5"
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch-39-LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch-40-PARITY="--bottom-drag-scheme nemo_quadratic"
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch-41-IC="--woa-init --woa-t $WTIC --woa-s $WSIC --sss-restore-file $SSSCLIM --ice-init $ICEIC"
--
scripts/cluster/omip_nemo/run_ico_ship1.sbatch-37-ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
scripts/cluster/omip_nemo/run_ico_ship1.sbatch-38-echo HOST $(hostname); date; git -C $WT log --oneline -1
scripts/cluster/omip_nemo/run_ico_ship1.sbatch-39-# GA/COMMON/LEVERS/IC identical to the ll_ship2 / trp2_ship1 protocol EXCEPT the
scripts/cluster/omip_nemo/run_ico_ship1.sbatch-40-# grid and the unavoidable MPAS gaps (no iwm/bbl/isf/dm2dc/drag-scheme/rgb-chl).
scripts/cluster/omip_nemo/run_ico_ship1.sbatch:41:GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
scripts/cluster/omip_nemo/run_ico_ship1.sbatch-42-COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.25 --snapshot-every-days 30 --diag-every-days 5"
scripts/cluster/omip_nemo/run_ico_ship1.sbatch-43-LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
scripts/cluster/omip_nemo/run_ico_ship1.sbatch-44-IC="--woa-init --woa-t $WTIC --woa-s $WSIC --sss-restore-file $SSSCLIM --ice-init $ICEIC"
scripts/cluster/omip_nemo/run_ico_ship1.sbatch-45-LABEL=ico_ship1
--
tests/unit/test_voronoi_precision.py-43-    def test_succeeds_without_x64(self):
tests/unit/test_voronoi_precision.py-44-        """Must succeed in float32 mode and produce float32 geometry."""
tests/unit/test_voronoi_precision.py-45-        code = """\
tests/unit/test_voronoi_precision.py-46-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_voronoi_precision.py:47:        m = create_voronoi_mesh(1, lloyd_iterations=2)
tests/unit/test_voronoi_precision.py-48-        assert m.areaCell.dtype.name == "float32", f"areaCell dtype: {m.areaCell.dtype}"
tests/unit/test_voronoi_precision.py-49-        print("OK")
tests/unit/test_voronoi_precision.py-50-        """
tests/unit/test_voronoi_precision.py-51-        rc, stdout, stderr = _run_snippet(code)
--
tests/unit/test_voronoi_precision.py-60-        code = """\
tests/unit/test_voronoi_precision.py-61-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_voronoi_precision.py-62-        import jax
tests/unit/test_voronoi_precision.py-63-        jax.config.update("jax_enable_x64", True)
tests/unit/test_voronoi_precision.py:64:        m = create_voronoi_mesh(1, lloyd_iterations=2)
tests/unit/test_voronoi_precision.py-65-        assert m.areaCell.dtype.name == "float64", f"areaCell dtype: {m.areaCell.dtype}"
tests/unit/test_voronoi_precision.py-66-        assert m.dvEdge.dtype.name == "float64", f"dvEdge dtype: {m.dvEdge.dtype}"
tests/unit/test_voronoi_precision.py-67-        assert m.weightsOnEdge.dtype.name == "float64", f"weightsOnEdge dtype: {m.weightsOnEdge.dtype}"
tests/unit/test_voronoi_precision.py-68-        print("OK")
--
tests/unit/test_voronoi_precision.py-79-    def test_env_var_enables_x64(self):
tests/unit/test_voronoi_precision.py-80-        """JAX_ENABLE_X64=1 env var should produce float64 geometry."""
tests/unit/test_voronoi_precision.py-81-        code = """\
tests/unit/test_voronoi_precision.py-82-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_voronoi_precision.py:83:        m = create_voronoi_mesh(1, lloyd_iterations=2)
tests/unit/test_voronoi_precision.py-84-        print(m.areaCell.dtype)
tests/unit/test_voronoi_precision.py-85-        """
tests/unit/test_voronoi_precision.py-86-        rc, stdout, stderr = _run_snippet(
tests/unit/test_voronoi_precision.py-87-            code, env_extra={"JAX_ENABLE_X64": "1"},
--
tests/unit/test_sea_ice_new_physics.py-1081-    """Ice physics on MPAS Voronoi mesh."""
tests/unit/test_sea_ice_new_physics.py-1082-
tests/unit/test_sea_ice_new_physics.py-1083-    def _make_mesh(self, level=2):
tests/unit/test_sea_ice_new_physics.py-1084-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_sea_ice_new_physics.py:1085:        return create_voronoi_mesh(subdivision_level=level, lloyd_iterations=5)
tests/unit/test_sea_ice_new_physics.py-1086-
tests/unit/test_sea_ice_new_physics.py-1087-    def test_transport_uniform_field_conserved(self):
tests/unit/test_sea_ice_new_physics.py-1088-        from legoesm.ice.transport import advect_ice_tracers
tests/unit/test_sea_ice_new_physics.py-1089-        mesh = self._make_mesh()
--
tests/unit/test_mpas_physics_ledger.py-118-    what the model actually applies, a term could hide in the gap.
tests/unit/test_mpas_physics_ledger.py-119-    """
tests/unit/test_mpas_physics_ledger.py-120-    # A REAL Voronoi mesh: Louis turbulence reconstructs cell velocity from
tests/unit/test_mpas_physics_ledger.py-121-    # edge normals (voronoi.reconstruct_cell_velocity), so the stub mesh is
tests/unit/test_mpas_physics_ledger.py:122:    # not enough.  Level 2 = 162 cells, and lloyd_iterations=0 keeps it fast
tests/unit/test_mpas_physics_ledger.py-123-    # (mesh QUALITY is irrelevant here -- this asserts an algebraic identity
tests/unit/test_mpas_physics_ledger.py-124-    # between rows and their sum, not a physical result).
tests/unit/test_mpas_physics_ledger.py-125-    from legoesm.core.state import Field, HydrostaticState
tests/unit/test_mpas_physics_ledger.py-126-    from legoesm.grids.vertical import create_sigma_coordinate
tests/unit/test_mpas_physics_ledger.py-127-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_mpas_physics_ledger.py-128-
tests/unit/test_mpas_physics_ledger.py:129:    mesh = create_voronoi_mesh(2, lloyd_iterations=0)
tests/unit/test_mpas_physics_ledger.py-130-    ncol = int(mesh.nCells)
tests/unit/test_mpas_physics_ledger.py-131-    nedge = int(np.asarray(mesh.dvEdge).size)
tests/unit/test_mpas_physics_ledger.py-132-    rng = np.random.default_rng(11)
tests/unit/test_mpas_physics_ledger.py-133-    sigma = create_sigma_coordinate(NLEV)
--
tests/unit/test_voronoi_to_latlon_regrid.py-19-
tests/unit/test_voronoi_to_latlon_regrid.py-20-@pytest.fixture(scope="module")
tests/unit/test_voronoi_to_latlon_regrid.py-21-def mesh():
tests/unit/test_voronoi_to_latlon_regrid.py-22-    # Level-3 SCVT mesh = 642 cells; cheap and enough to exercise the regrid.
tests/unit/test_voronoi_to_latlon_regrid.py:23:    return create_voronoi_mesh(3, lloyd_iterations=10)
tests/unit/test_voronoi_to_latlon_regrid.py-24-
tests/unit/test_voronoi_to_latlon_regrid.py-25-
tests/unit/test_voronoi_to_latlon_regrid.py-26-def test_weights_shapes_and_normalization(mesh):
tests/unit/test_voronoi_to_latlon_regrid.py-27-    w = compute_voronoi_to_latlon_weights(
--
tests/unit/test_physical_balances.py-177-            MPASShallowWaterState,
tests/unit/test_physical_balances.py-178-        )
tests/unit/test_physical_balances.py-179-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_physical_balances.py-180-
tests/unit/test_physical_balances.py:181:        self.mesh = create_voronoi_mesh(2, lloyd_iterations=30)
tests/unit/test_physical_balances.py-182-        mesh = self.mesh
tests/unit/test_physical_balances.py-183-
tests/unit/test_physical_balances.py-184-        g = constants.g
tests/unit/test_physical_balances.py-185-        Omega = constants.Omega
--
tests/unit/test_physical_balances.py-239-        nlev = 5
tests/unit/test_physical_balances.py-240-        T0 = 250.0
tests/unit/test_physical_balances.py-241-        ps0 = 1e5
tests/unit/test_physical_balances.py-242-
tests/unit/test_physical_balances.py:243:        self.mesh = create_voronoi_mesh(2, lloyd_iterations=30)
tests/unit/test_physical_balances.py-244-        self.sigma = create_sigma_coordinate(nlev)
tests/unit/test_physical_balances.py-245-        mesh = self.mesh
tests/unit/test_physical_balances.py-246-
tests/unit/test_physical_balances.py-247-        self.state0 = MPASHydrostaticState(
--
tests/unit/test_convergence_rates.py-370-    def _compute_gradient_error(level):
tests/unit/test_convergence_rates.py-371-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_convergence_rates.py-372-        from legoesm.core.operators_voronoi import gradient_edge
tests/unit/test_convergence_rates.py-373-
tests/unit/test_convergence_rates.py:374:        mesh = create_voronoi_mesh(level, lloyd_iterations=30)
tests/unit/test_convergence_rates.py-375-        a = mesh.radius
tests/unit/test_convergence_rates.py-376-
tests/unit/test_convergence_rates.py-377-        phi = jnp.cos(mesh.latCell) * jnp.cos(2.0 * mesh.lonCell)
tests/unit/test_convergence_rates.py-378-        grad = gradient_edge(phi, mesh)
--
tests/unit/test_convergence_rates.py-415-            MPASShallowWaterState,
tests/unit/test_convergence_rates.py-416-        )
tests/unit/test_convergence_rates.py-417-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_convergence_rates.py-418-
tests/unit/test_convergence_rates.py:419:        mesh = create_voronoi_mesh(level, lloyd_iterations=30)
tests/unit/test_convergence_rates.py-420-
tests/unit/test_convergence_rates.py-421-        g = constants.g
tests/unit/test_convergence_rates.py-422-        Omega = constants.Omega
tests/unit/test_convergence_rates.py-423-        R = mesh.radius
--
tests/unit/test_voronoi_trisk_weights.py-31-def mesh(request):
tests/unit/test_voronoi_trisk_weights.py-32-    """SCVT meshes at three resolutions to stress the formula over
tests/unit/test_voronoi_trisk_weights.py-33-    varying kite-area asymmetry."""
tests/unit/test_voronoi_trisk_weights.py-34-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_voronoi_trisk_weights.py:35:    level, lloyd = request.param
tests/unit/test_voronoi_trisk_weights.py:36:    return create_voronoi_mesh(level, lloyd_iterations=lloyd)
tests/unit/test_voronoi_trisk_weights.py-37-
tests/unit/test_voronoi_trisk_weights.py-38-
tests/unit/test_voronoi_trisk_weights.py-39-@pytest.fixture(scope="module")
tests/unit/test_voronoi_trisk_weights.py-40-def channel_mesh():
--
tests/unit/test_voronoi_partition_method.py-25-    partition_cells_geometric,
tests/unit/test_voronoi_partition_method.py-26-    partition_cells_metis,
tests/unit/test_voronoi_partition_method.py-27-    partition_cells_sfc,
tests/unit/test_voronoi_partition_method.py-28-    partition_voronoi_mesh,
tests/unit/test_voronoi_partition_method.py:29:    reorder_voronoi_for_sharding,
tests/unit/test_voronoi_partition_method.py-30-    resolve_partition_method,
tests/unit/test_voronoi_partition_method.py-31-)
tests/unit/test_voronoi_partition_method.py-32-
tests/unit/test_voronoi_partition_method.py-33-N_RANKS = 4
--
tests/unit/test_voronoi_partition_method.py-35-
tests/unit/test_voronoi_partition_method.py-36-@pytest.fixture(scope="module")
tests/unit/test_voronoi_partition_method.py-37-def mesh():
tests/unit/test_voronoi_partition_method.py-38-    # Level-1 SCVT: 10*4^1 + 2 = 42 cells. Tiny + fast.
tests/unit/test_voronoi_partition_method.py:39:    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
tests/unit/test_voronoi_partition_method.py-40-
tests/unit/test_voronoi_partition_method.py-41-
tests/unit/test_voronoi_partition_method.py-42-def _assert_valid_owner(owner, n_cells, n_ranks):
tests/unit/test_voronoi_partition_method.py-43-    owner = np.asarray(owner)
--
tests/unit/test_voronoi_partition_method.py-157-            partition_voronoi_mesh(mesh, N_RANKS, 0, method="bogus")
tests/unit/test_voronoi_partition_method.py-158-
tests/unit/test_voronoi_partition_method.py-159-    def test_reorder_rejects_unknown(self, mesh):
tests/unit/test_voronoi_partition_method.py-160-        with pytest.raises(ValueError, match="Unknown partitioning method"):
tests/unit/test_voronoi_partition_method.py:161:            reorder_voronoi_for_sharding(mesh, 2, method="bogus")
tests/unit/test_voronoi_partition_method.py-162-
tests/unit/test_voronoi_partition_method.py-163-    def test_reorder_single_device_still_rejects_unknown(self, mesh):
tests/unit/test_voronoi_partition_method.py-164-        # Validate-at-entry: the n_devices<=1 shortcut must not skip the guard.
tests/unit/test_voronoi_partition_method.py-165-        with pytest.raises(ValueError, match="Unknown partitioning method"):
tests/unit/test_voronoi_partition_method.py:166:            reorder_voronoi_for_sharding(mesh, 1, method="bogus")
tests/unit/test_voronoi_partition_method.py-167-
tests/unit/test_voronoi_partition_method.py-168-    def test_partition_with_cell_owner_still_rejects_unknown(self, mesh):
tests/unit/test_voronoi_partition_method.py-169-        # Validate-at-entry: supplying cell_owner must not skip the guard.
tests/unit/test_voronoi_partition_method.py-170-        owner = np.zeros(mesh.nCells, dtype=np.int32)
--
tests/unit/test_voronoi_partition_method.py-186-        assert np.array_equal(auto.cell_g2l, geom.cell_g2l)
tests/unit/test_voronoi_partition_method.py-187-
tests/unit/test_voronoi_partition_method.py-188-    def test_reorder_auto_runs_without_metis(self, mesh, monkeypatch):
tests/unit/test_voronoi_partition_method.py-189-        monkeypatch.setattr(vp, "_metis_available", lambda: False)
tests/unit/test_voronoi_partition_method.py:190:        reordered = reorder_voronoi_for_sharding(mesh, 2, method="auto")
tests/unit/test_voronoi_partition_method.py-191-        assert reordered.nCells == mesh.nCells
tests/unit/test_voronoi_partition_method.py-192-
tests/unit/test_voronoi_partition_method.py-193-    def test_reorder_single_device_noop(self, mesh):
tests/unit/test_voronoi_partition_method.py-194-        # n_devices <= 1 returns the mesh unchanged (no partition needed).
tests/unit/test_voronoi_partition_method.py:195:        assert reorder_voronoi_for_sharding(mesh, 1, method="auto") is mesh
tests/unit/test_voronoi_partition_method.py-196-
tests/unit/test_voronoi_partition_method.py-197-
tests/unit/test_voronoi_partition_method.py-198-# ---------------------------------------------------------------------------
tests/unit/test_voronoi_partition_method.py-199-# Owned-first local indexing (existing contract — locked by a test)
--
tests/unit/test_voronoi_partition_method.py-226-        part = partition_voronoi_mesh(mesh, N_RANKS, 0, method="sfc")
tests/unit/test_voronoi_partition_method.py-227-        assert part.n_owned_cells > 0
tests/unit/test_voronoi_partition_method.py-228-
tests/unit/test_voronoi_partition_method.py-229-    def test_reorder_sfc_is_permutation(self, mesh):
tests/unit/test_voronoi_partition_method.py:230:        reordered = reorder_voronoi_for_sharding(mesh, 2, method="sfc")
tests/unit/test_voronoi_partition_method.py-231-        assert reordered.nCells == mesh.nCells
tests/unit/test_voronoi_partition_method.py-232-        # Reorder is a pure permutation: the cell-latitude multiset is preserved.
tests/unit/test_voronoi_partition_method.py-233-        assert np.allclose(
tests/unit/test_voronoi_partition_method.py-234-            np.sort(np.asarray(reordered.latCell)),
--
tests/unit/test_voronoi_partition_method.py-236-        )
tests/unit/test_voronoi_partition_method.py-237-
tests/unit/test_voronoi_partition_method.py-238-    def test_reorder_default_preserves_cells(self, mesh):
tests/unit/test_voronoi_partition_method.py-239-        # Default method (auto->geometric here) now SFC-orders within owners.
tests/unit/test_voronoi_partition_method.py:240:        reordered = reorder_voronoi_for_sharding(mesh, 2)
tests/unit/test_voronoi_partition_method.py-241-        assert reordered.nCells == mesh.nCells
tests/unit/test_voronoi_partition_method.py-242-
tests/unit/test_voronoi_partition_method.py-243-
tests/unit/test_voronoi_partition_method.py-244-def test_sharding_reorder_auto_is_sfc_not_metis(monkeypatch):
--
tests/unit/test_voronoi_partition_method.py-247-    The global ``resolve_partition_method`` policy prefers METIS when pymetis
tests/unit/test_voronoi_partition_method.py-248-    is importable, which optimizes EDGE CUT. The cost that binds this path at
tests/unit/test_voronoi_partition_method.py-249-    high device counts is the collective-permute ROUND count (= max degree of
tests/unit/test_voronoi_partition_method.py-250-    the post-reorder depth-3+closure comm graph). Measured on the real layout
tests/unit/test_voronoi_partition_method.py:251:    those objectives move oppositely -- subdiv-8 @128: sfc 14 rounds, metis 19
tests/unit/test_voronoi_partition_method.py-252-    -- so auto must NOT inherit the METIS preference here.
tests/unit/test_voronoi_partition_method.py-253-
tests/unit/test_voronoi_partition_method.py-254-    Asserted behaviourally, by which partitioner the reorder actually calls:
tests/unit/test_voronoi_partition_method.py-255-    a check on the resolver alone would prove nothing about this path.
--
tests/unit/test_voronoi_partition_method.py-273-            return _real(m, n)
tests/unit/test_voronoi_partition_method.py-274-
tests/unit/test_voronoi_partition_method.py-275-        monkeypatch.setattr(vp, name, _tap)
tests/unit/test_voronoi_partition_method.py-276-
tests/unit/test_voronoi_partition_method.py:277:    vp.reorder_voronoi_for_sharding(create_voronoi_mesh(subdivision_level=3), 4)
tests/unit/test_voronoi_partition_method.py-278-
tests/unit/test_voronoi_partition_method.py-279-    assert used == ["partition_cells_sfc"], (
tests/unit/test_voronoi_partition_method.py:280:        f"reorder_voronoi_for_sharding(method='auto') used {used}; it must "
tests/unit/test_voronoi_partition_method.py-281-        f"use SFC — METIS costs +15 collective-permutes/step at 128 devices "
tests/unit/test_voronoi_partition_method.py:282:        f"on subdiv-8/9.")
tests/unit/test_voronoi_partition_method.py-283-
tests/unit/test_voronoi_partition_method.py-284-
tests/unit/test_voronoi_partition_method.py-285-def test_sharding_reorder_still_honours_an_explicit_method(monkeypatch):
tests/unit/test_voronoi_partition_method.py-286-    """The auto override must not hijack an EXPLICIT choice — a deck that
--
tests/unit/test_voronoi_partition_method.py-304-                used.append(_name)
tests/unit/test_voronoi_partition_method.py-305-                return _real(m, n)
tests/unit/test_voronoi_partition_method.py-306-
tests/unit/test_voronoi_partition_method.py-307-            monkeypatch.setattr(vp, name, _tap)
tests/unit/test_voronoi_partition_method.py:308:        vp.reorder_voronoi_for_sharding(mesh, 4, method=method)
tests/unit/test_voronoi_partition_method.py-309-        assert used == [expect], f"method={method!r} used {used}"
--
tests/unit/test_voronoi_surface_fields.py-23-
tests/unit/test_voronoi_surface_fields.py-24-
tests/unit/test_voronoi_surface_fields.py-25-@pytest.fixture(scope="module")
tests/unit/test_voronoi_surface_fields.py-26-def mesh():
tests/unit/test_voronoi_surface_fields.py:27:    return create_grid("mpas", 2, lloyd_iterations=5)   # 162 cells, fast
tests/unit/test_voronoi_surface_fields.py-28-
tests/unit/test_voronoi_surface_fields.py-29-
tests/unit/test_voronoi_surface_fields.py-30-def test_fields_default_none_and_extractors_fall_back(mesh):
tests/unit/test_voronoi_surface_fields.py-31-    assert mesh.subgrid_topo_stddev is None
--
tests/unit/test_voronoi_surface_fields.py-104-    from legoesm.grids.voronoi import (
tests/unit/test_voronoi_surface_fields.py-105-        _load_voronoi_cache, _save_voronoi_cache,
tests/unit/test_voronoi_surface_fields.py-106-    )
tests/unit/test_voronoi_surface_fields.py-107-
tests/unit/test_voronoi_surface_fields.py:108:    m = create_grid("mpas", 2, lloyd_iterations=5)
tests/unit/test_voronoi_surface_fields.py-109-    assert m.subgrid_topo_stddev is None          # the None-optional case
tests/unit/test_voronoi_surface_fields.py-110-    p = str(tmp_path / "mesh_cache.npz")
tests/unit/test_voronoi_surface_fields.py-111-    with _w.catch_warnings():
tests/unit/test_voronoi_surface_fields.py-112-        _w.simplefilter("error")                  # a cache-write warning FAILS
--
tests/unit/test_mpas_step_ledger.py-44-    )
tests/unit/test_mpas_step_ledger.py-45-    from legoesm.grids.vertical import create_sigma_coordinate
tests/unit/test_mpas_step_ledger.py-46-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_mpas_step_ledger.py-47-
tests/unit/test_mpas_step_ledger.py:48:    mesh = create_voronoi_mesh(2, lloyd_iterations=0)
tests/unit/test_mpas_step_ledger.py-49-    sigma = create_sigma_coordinate(NLEV)
tests/unit/test_mpas_step_ledger.py-50-    cfg = MPASPrimitiveEquationConfig(
tests/unit/test_mpas_step_ledger.py-51-        conservative_tracer_clamp=conservative_clamp)
tests/unit/test_mpas_step_ledger.py-52-    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
--
tests/unit/test_mpas_atmosphere.py-48-# ============================================================================
tests/unit/test_mpas_atmosphere.py-49-
tests/unit/test_mpas_atmosphere.py-50-def _make_mesh(level=2):
tests/unit/test_mpas_atmosphere.py-51-    """Small icosahedral mesh for testing (level 2 = 162 cells)."""
tests/unit/test_mpas_atmosphere.py:52:    return create_voronoi_mesh(level, lloyd_iterations=5)
tests/unit/test_mpas_atmosphere.py-53-
tests/unit/test_mpas_atmosphere.py-54-
tests/unit/test_mpas_atmosphere.py-55-def _make_hydrostatic_state(mesh, nlev=5):
tests/unit/test_mpas_atmosphere.py-56-    """Create a simple rest state for the hydrostatic PE."""
--
tests/unit/test_mpas_atmosphere.py-270-
tests/unit/test_mpas_atmosphere.py-271-    @classmethod
tests/unit/test_mpas_atmosphere.py-272-    def setUpClass(cls):
tests/unit/test_mpas_atmosphere.py-273-        cls.nlev = 30
tests/unit/test_mpas_atmosphere.py:274:        cls.mesh = create_voronoi_mesh(4, lloyd_iterations=5)
tests/unit/test_mpas_atmosphere.py-275-        cls.sigma = create_sigma_coordinate(cls.nlev)
tests/unit/test_mpas_atmosphere.py-276-        # Operational hyperdiffusion coefficients — the regime in which the
tests/unit/test_mpas_atmosphere.py-277-        # integrator choice matters.  Same del2/del4 scaling as the GPU
tests/unit/test_mpas_atmosphere.py-278-        # integrator sweep (``_diag_mpas_integrator_sweep.py``).
--
tests/unit/test_mpas_atmosphere.py-411-        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
tests/unit/test_mpas_atmosphere.py-412-            GravityWaveDragConfig,
tests/unit/test_mpas_atmosphere.py-413-        )
tests/unit/test_mpas_atmosphere.py-414-        cls.nlev = 20
tests/unit/test_mpas_atmosphere.py:415:        cls.mesh = create_voronoi_mesh(3, lloyd_iterations=3)
tests/unit/test_mpas_atmosphere.py-416-        cls.sigma = create_sigma_coordinate(cls.nlev)
tests/unit/test_mpas_atmosphere.py-417-        cls.state = _make_hydrostatic_state(cls.mesh, cls.nlev)
tests/unit/test_mpas_atmosphere.py-418-        cls.ncell = cls.mesh.nCells
tests/unit/test_mpas_atmosphere.py-419-        # gray radiation only; everything else off (mirrors the MPAS AMIP run
--
tests/unit/test_mpas_atmosphere.py-476-
tests/unit/test_mpas_atmosphere.py-477-    @classmethod
tests/unit/test_mpas_atmosphere.py-478-    def setUpClass(cls):
tests/unit/test_mpas_atmosphere.py-479-        cls.nlev = 16
tests/unit/test_mpas_atmosphere.py:480:        cls.mesh = create_voronoi_mesh(3, lloyd_iterations=3)
tests/unit/test_mpas_atmosphere.py-481-        cls.sigma = create_sigma_coordinate(cls.nlev)
tests/unit/test_mpas_atmosphere.py-482-        cls.model = MPASPrimitiveEquationModel(
tests/unit/test_mpas_atmosphere.py-483-            cls.mesh, cls.sigma, MPASPrimitiveEquationConfig())
tests/unit/test_mpas_atmosphere.py-484-
--
tests/unit/test_mpas_atmosphere.py-536-
tests/unit/test_mpas_atmosphere.py-537-    @classmethod
tests/unit/test_mpas_atmosphere.py-538-    def setUpClass(cls):
tests/unit/test_mpas_atmosphere.py-539-        cls.nlev = 12
tests/unit/test_mpas_atmosphere.py:540:        cls.mesh = create_voronoi_mesh(3, lloyd_iterations=3)
tests/unit/test_mpas_atmosphere.py-541-        cls.sigma = create_sigma_coordinate(cls.nlev)
tests/unit/test_mpas_atmosphere.py-542-        cls.state = _add_perturbation_hydro(
tests/unit/test_mpas_atmosphere.py-543-            _make_hydrostatic_state(cls.mesh, cls.nlev), cls.mesh, cls.nlev)
tests/unit/test_mpas_atmosphere.py-544-        cls.model = MPASPrimitiveEquationModel(
--
tests/unit/test_sea_ice_dynamics.py-3060-        self._run_grid(grid, (16, 32), "evp")
tests/unit/test_sea_ice_dynamics.py-3061-
tests/unit/test_sea_ice_dynamics.py-3062-    def test_mpas_voronoi_coupled(self):
tests/unit/test_sea_ice_dynamics.py-3063-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_sea_ice_dynamics.py:3064:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/unit/test_sea_ice_dynamics.py-3065-        self._run_grid(mesh, (mesh.nCells,), "free_drift")
tests/unit/test_sea_ice_dynamics.py-3066-
tests/unit/test_sea_ice_dynamics.py-3067-
tests/unit/test_sea_ice_dynamics.py-3068-class TestVoronoiTransportUpwind:
--
tests/unit/test_sea_ice_dynamics.py-3075-    def test_upwind_preserves_positivity(self):
tests/unit/test_sea_ice_dynamics.py-3076-        import numpy as np
tests/unit/test_sea_ice_dynamics.py-3077-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_sea_ice_dynamics.py-3078-        from legoesm.ice.transport import fv_flux_divergence_voronoi
tests/unit/test_sea_ice_dynamics.py:3079:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/unit/test_sea_ice_dynamics.py-3080-        nC = int(mesh.nCells)
tests/unit/test_sea_ice_dynamics.py-3081-        # Sharp step field in [0,1] -> strong gradients at every interface.
tests/unit/test_sea_ice_dynamics.py-3082-        q = jnp.asarray((np.arange(nC) % 2).astype(np.float64))
tests/unit/test_sea_ice_dynamics.py-3083-        u = jnp.ones((nC,)); v = jnp.zeros((nC,))   # uniform east, |u_edge|<=1
--
tests/unit/test_voronoi_mesh_cache.py-1-"""Tests for the Voronoi (MPAS) mesh disk cache.
tests/unit/test_voronoi_mesh_cache.py-2-
tests/unit/test_voronoi_mesh_cache.py-3-The SCVT build (Lloyd relaxation) is deterministic in
tests/unit/test_voronoi_mesh_cache.py:4:``(subdivision_level, radius, lloyd_iterations, omega)`` when ``density_fn`` is
tests/unit/test_voronoi_mesh_cache.py-5-None, but slow (~10^2 s at production resolution).  Under MPI every rank rebuilds
tests/unit/test_voronoi_mesh_cache.py-6-the same mesh, which at high rank-count contends for cores and stalls the job.
tests/unit/test_voronoi_mesh_cache.py-7-``create_voronoi_mesh`` therefore caches the built mesh to a ``.npz`` keyed on
tests/unit/test_voronoi_mesh_cache.py-8-those args and loads it on a hit.  These tests exercise: round-trip fidelity, the
--
tests/unit/test_voronoi_mesh_cache.py-58-    """First call builds + writes the .npz; second call loads an identical mesh."""
tests/unit/test_voronoi_mesh_cache.py-59-    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
tests/unit/test_voronoi_mesh_cache.py-60-    assert not os.path.exists(path)
tests/unit/test_voronoi_mesh_cache.py-61-
tests/unit/test_voronoi_mesh_cache.py:62:    m1 = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
tests/unit/test_voronoi_mesh_cache.py-63-    assert os.path.exists(path), "build did not write the cache file"
tests/unit/test_voronoi_mesh_cache.py-64-
tests/unit/test_voronoi_mesh_cache.py:65:    m2 = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
tests/unit/test_voronoi_mesh_cache.py-66-    _mesh_fields_equal(m1, m2)
tests/unit/test_voronoi_mesh_cache.py-67-
tests/unit/test_voronoi_mesh_cache.py-68-
tests/unit/test_voronoi_mesh_cache.py-69-def test_loaded_mesh_matches_uncached(cache_dir, monkeypatch):
tests/unit/test_voronoi_mesh_cache.py-70-    """A cache-loaded mesh equals the same mesh built with the cache disabled."""
tests/unit/test_voronoi_mesh_cache.py:71:    cached = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
tests/unit/test_voronoi_mesh_cache.py-72-
tests/unit/test_voronoi_mesh_cache.py-73-    monkeypatch.setenv("LEGOESM_MESH_CACHE_DISABLE", "1")
tests/unit/test_voronoi_mesh_cache.py-74-    assert _voronoi_cache_disabled()
tests/unit/test_voronoi_mesh_cache.py:75:    fresh = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
tests/unit/test_voronoi_mesh_cache.py-76-    _mesh_fields_equal(cached, fresh)
tests/unit/test_voronoi_mesh_cache.py-77-
tests/unit/test_voronoi_mesh_cache.py-78-
tests/unit/test_voronoi_mesh_cache.py-79-def test_disable_env_skips_write(cache_dir, monkeypatch):
tests/unit/test_voronoi_mesh_cache.py-80-    """With the cache disabled, no file is written and nothing is loaded."""
tests/unit/test_voronoi_mesh_cache.py-81-    monkeypatch.setenv("LEGOESM_MESH_CACHE_DISABLE", "1")
tests/unit/test_voronoi_mesh_cache.py-82-    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
tests/unit/test_voronoi_mesh_cache.py:83:    create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
tests/unit/test_voronoi_mesh_cache.py-84-    assert not os.path.exists(path), "cache file written despite DISABLE=1"
tests/unit/test_voronoi_mesh_cache.py-85-
tests/unit/test_voronoi_mesh_cache.py-86-
tests/unit/test_voronoi_mesh_cache.py-87-def test_key_separates_params(cache_dir):
--
tests/unit/test_voronoi_mesh_cache.py-122-
tests/unit/test_voronoi_mesh_cache.py-123-
tests/unit/test_voronoi_mesh_cache.py-124-def test_save_is_atomic_no_tmp_left(cache_dir):
tests/unit/test_voronoi_mesh_cache.py-125-    """After a successful save, only the final file remains (no .tmp.<pid> debris)."""
tests/unit/test_voronoi_mesh_cache.py:126:    m = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
tests/unit/test_voronoi_mesh_cache.py-127-    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
tests/unit/test_voronoi_mesh_cache.py-128-    # Re-save explicitly and confirm no temp files linger in the dir.
tests/unit/test_voronoi_mesh_cache.py-129-    _save_voronoi_cache(path, m)
tests/unit/test_voronoi_mesh_cache.py-130-    leftovers = [p for p in os.listdir(os.path.dirname(path)) if ".tmp." in p]
--
tests/unit/test_voronoi_mesh_cache.py-141-
tests/unit/test_voronoi_mesh_cache.py-142-
tests/unit/test_voronoi_mesh_cache.py-143-def test_prewarm_creates_cache(cache_dir):
tests/unit/test_voronoi_mesh_cache.py-144-    """prewarm_voronoi_cache builds + caches and returns the existing path."""
tests/unit/test_voronoi_mesh_cache.py:145:    path = prewarm_voronoi_cache(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
tests/unit/test_voronoi_mesh_cache.py-146-    assert os.path.exists(path)
tests/unit/test_voronoi_mesh_cache.py-147-    assert path == _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
tests/unit/test_voronoi_mesh_cache.py-148-
tests/unit/test_voronoi_mesh_cache.py-149-
tests/unit/test_voronoi_mesh_cache.py-150-def test_prewarm_refuses_when_disabled(cache_dir, monkeypatch):
tests/unit/test_voronoi_mesh_cache.py-151-    """Pre-warming a disabled cache is a hard error (no silent no-op)."""
tests/unit/test_voronoi_mesh_cache.py-152-    monkeypatch.setenv("LEGOESM_MESH_CACHE_DISABLE", "1")
tests/unit/test_voronoi_mesh_cache.py-153-    with pytest.raises(RuntimeError, match="disabled cache"):
tests/unit/test_voronoi_mesh_cache.py:154:        prewarm_voronoi_cache(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
--
tests/unit/test_mpas_cmip_accumulator_feed.py-38-@pytest.fixture(scope="module")
tests/unit/test_mpas_cmip_accumulator_feed.py-39-def mesh():
tests/unit/test_mpas_cmip_accumulator_feed.py-40-    # Level-2 SCVT mesh = 162 cells / 480 edges; cheap, enough to exercise
tests/unit/test_mpas_cmip_accumulator_feed.py-41-    # both the IDW regrid and the Perot edge->cell wind reconstruction.
tests/unit/test_mpas_cmip_accumulator_feed.py:42:    return create_grid("mpas", 2, lloyd_iterations=10)
tests/unit/test_mpas_cmip_accumulator_feed.py-43-
tests/unit/test_mpas_cmip_accumulator_feed.py-44-
tests/unit/test_mpas_cmip_accumulator_feed.py-45-def _make_collector(mesh, *, monthly_means=True, cmip_output=True,
tests/unit/test_mpas_cmip_accumulator_feed.py-46-                    cmip_resolution_deg=10.0, output_dir=None):
--
tests/unit/test_grid_factory.py-28-    [
tests/unit/test_grid_factory.py-29-        ("cubed_sphere", 4, {}, "CubedSphereGrid"),
tests/unit/test_grid_factory.py-30-        ("gaussian", 21, {}, "GaussianGrid"),
tests/unit/test_grid_factory.py-31-        ("latlon", 16, {}, "LatLonGrid"),
tests/unit/test_grid_factory.py:32:        ("mpas", 1, {"lloyd_iterations": 2}, "VoronoiMesh"),
tests/unit/test_grid_factory.py-33-    ],
tests/unit/test_grid_factory.py-34-)
tests/unit/test_grid_factory.py-35-def test_create_each_global_grid(grid_type, resolution, kwargs, expected_cls) -> None:
tests/unit/test_grid_factory.py-36-    """Every global grid ocean and atmosphere share instantiates via one entry."""
--
tests/unit/test_radiation_number_coupling.py-157-    )
tests/unit/test_radiation_number_coupling.py-158-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_radiation_number_coupling.py-159-
tests/unit/test_radiation_number_coupling.py-160-    nlev = 8
tests/unit/test_radiation_number_coupling.py:161:    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
tests/unit/test_radiation_number_coupling.py-162-    hc = create_height_coordinate(nlev, 20_000.0)
tests/unit/test_radiation_number_coupling.py-163-    tm = compute_terrain_metric(jnp.zeros((mesh.nCells,)), hc)
tests/unit/test_radiation_number_coupling.py-164-    dc = ("nCells", "nlev")
tests/unit/test_radiation_number_coupling.py-165-    dw = ("nCells", "nlev_half")
--
tests/unit/test_tracer_transport_mpas.py-67-# ============================================================================
tests/unit/test_tracer_transport_mpas.py-68-
tests/unit/test_tracer_transport_mpas.py-69-@pytest.fixture(scope="module")
tests/unit/test_tracer_transport_mpas.py-70-def mesh():
tests/unit/test_tracer_transport_mpas.py:71:    return create_voronoi_mesh(LEVEL, lloyd_iterations=5)
tests/unit/test_tracer_transport_mpas.py-72-
tests/unit/test_tracer_transport_mpas.py-73-
tests/unit/test_tracer_transport_mpas.py-74-@pytest.fixture(scope="module")
tests/unit/test_tracer_transport_mpas.py-75-def sigma():
--
tests/unit/test_mpas_cmor_flux_feed.py-23-
tests/unit/test_mpas_cmor_flux_feed.py-24-
tests/unit/test_mpas_cmor_flux_feed.py-25-@pytest.fixture(scope="module")
tests/unit/test_mpas_cmor_flux_feed.py-26-def mesh():
tests/unit/test_mpas_cmor_flux_feed.py:27:    return create_grid("mpas", 2, lloyd_iterations=10)
tests/unit/test_mpas_cmor_flux_feed.py-28-
tests/unit/test_mpas_cmor_flux_feed.py-29-
tests/unit/test_mpas_cmor_flux_feed.py-30-def _make_collector(mesh):
tests/unit/test_mpas_cmor_flux_feed.py-31-    sigma_full = np.linspace(0.05, 0.98, NLEV)
--
tests/unit/test_mpas_land_boundary.py-136-
tests/unit/test_mpas_land_boundary.py-137-@pytest.fixture(scope="module")
tests/unit/test_mpas_land_boundary.py-138-def mpas_mesh():
tests/unit/test_mpas_land_boundary.py-139-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_mpas_land_boundary.py:140:    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
tests/unit/test_mpas_land_boundary.py-141-
tests/unit/test_mpas_land_boundary.py-142-
tests/unit/test_mpas_land_boundary.py-143-@pytest.fixture(scope="module")
tests/unit/test_mpas_land_boundary.py-144-def sigma_coord():
--
tests/unit/test_prewarm_voronoi_mesh_cli.py-28-
tests/unit/test_prewarm_voronoi_mesh_cli.py-29-
tests/unit/test_prewarm_voronoi_mesh_cli.py-30-def test_prewarms_small_level(tmp_path):
tests/unit/test_prewarm_voronoi_mesh_cli.py-31-    r = subprocess.run(
tests/unit/test_prewarm_voronoi_mesh_cli.py:32:        [sys.executable, str(SCRIPT), "--level", "3", "--lloyd", "0"],
tests/unit/test_prewarm_voronoi_mesh_cli.py-33-        env=_env(tmp_path), capture_output=True, text=True)
tests/unit/test_prewarm_voronoi_mesh_cli.py-34-    assert r.returncode == 0, r.stderr
tests/unit/test_prewarm_voronoi_mesh_cli.py-35-    assert "prewarmed level=3" in r.stdout
tests/unit/test_prewarm_voronoi_mesh_cli.py-36-    assert any(p.suffix == ".npz" for p in tmp_path.rglob("*.npz"))
--
tests/unit/test_mpas_clt_feed.py-30-
tests/unit/test_mpas_clt_feed.py-31-@pytest.fixture(scope="module")
tests/unit/test_mpas_clt_feed.py-32-def mesh():
tests/unit/test_mpas_clt_feed.py-33-    # Level-2 SCVT mesh = 162 cells; cheap, exercises the real IDW regrid.
tests/unit/test_mpas_clt_feed.py:34:    return create_grid("mpas", 2, lloyd_iterations=10)
tests/unit/test_mpas_clt_feed.py-35-
tests/unit/test_mpas_clt_feed.py-36-
tests/unit/test_mpas_clt_feed.py-37-def _collector(mesh, cloud_config):
tests/unit/test_mpas_clt_feed.py-38-    sigma_full = np.linspace(0.05, 0.98, NLEV)
--
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py-72-def mpas_mesh():
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py-73-    # Smallest viable mesh: subdivision_level=1 → 42 cells.  Level 2
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py-74-    # (162 cells) is needed for some dycore operators but the column-
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py-75-    # wise convection bridge is fine at level 1.
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:76:    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py-77-
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py-78-
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py-79-@pytest.fixture(scope="module")
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py-80-def sigma_coord():
--
tests/unit/test_symmetry_invariance.py-159-            MPASShallowWaterState,
tests/unit/test_symmetry_invariance.py-160-        )
tests/unit/test_symmetry_invariance.py-161-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_symmetry_invariance.py-162-
tests/unit/test_symmetry_invariance.py:163:        mesh = create_voronoi_mesh(2, lloyd_iterations=30)
tests/unit/test_symmetry_invariance.py-164-        config = MPASShallowWaterConfig(fix_mass=True, fix_energy=False)
tests/unit/test_symmetry_invariance.py-165-        model = MPASShallowWaterModel(mesh, config)
tests/unit/test_symmetry_invariance.py-166-        return model, mesh
tests/unit/test_symmetry_invariance.py-167-
--
tests/unit/test_conservation_laws.py-129-        )
tests/unit/test_conservation_laws.py-130-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_conservation_laws.py-131-        from legoesm.core.operators_voronoi import curl_vertex, vertex_thickness
tests/unit/test_conservation_laws.py-132-
tests/unit/test_conservation_laws.py:133:        mesh = create_voronoi_mesh(2, lloyd_iterations=30)
tests/unit/test_conservation_laws.py-134-
tests/unit/test_conservation_laws.py-135-        g = constants.g
tests/unit/test_conservation_laws.py-136-        Omega = constants.Omega
tests/unit/test_conservation_laws.py-137-        R = constants.R_earth
--
tests/unit/test_conservation_laws.py-207-        )
tests/unit/test_conservation_laws.py-208-        from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_conservation_laws.py-209-        from legoesm.core.operators_voronoi import kinetic_energy_cell
tests/unit/test_conservation_laws.py-210-
tests/unit/test_conservation_laws.py:211:        self.mesh = create_voronoi_mesh(2, lloyd_iterations=30)
tests/unit/test_conservation_laws.py-212-        mesh = self.mesh
tests/unit/test_conservation_laws.py-213-        self.kinetic_energy_cell = kinetic_energy_cell
tests/unit/test_conservation_laws.py-214-
tests/unit/test_conservation_laws.py-215-        g = constants.g

exec
/usr/bin/bash -lc 'rg -n "''^(def|class) (_build_ppermute_schedule|spmd_schedule_cost|reorder_voronoi_for_sharding)|SPMD_HALO_DEPTH|reference census|subdiv-8.*sfc|subdiv-9.*sfc" legoesm/parallel legoesm/grids scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch && nl -ba legoesm/parallel/sharded_dynamics.py | sed -n '"'1,420p' && nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch | sed -n '1,300p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 2 in 97ms:
rg: legoesm/parallel: No such file or directory (os error 2)
rg: legoesm/grids: No such file or directory (os error 2)
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:44:# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:45:#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:46:#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:51:# lloyd=0 throughout: it is what the reference census used AND the cache key

exec
/usr/bin/bash -lc "pwd; rg --files -g 'sharded_dynamics.py' -g 'voronoi_partition.py' -g 'voronoi.py' -g 'bench_voronoi_partition_methods.py' -g 'mpas_schedule_cost_scan.sbatch' -g 'AGENTS.md' | sort; find . -maxdepth 3 -type d -name 'legoesm' -print; nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch | sed -n '1,260p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 541ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
packages/core/legoesm/grids/voronoi.py
packages/core/legoesm/parallel/sharded_dynamics.py
packages/core/legoesm/parallel/voronoi_partition.py
scripts/bench/bench_voronoi_partition_methods.py
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
./src/legoesm
./packages/ice/legoesm
./packages/atmosphere/legoesm
./packages/ml/legoesm
./packages/coupler/legoesm
./packages/land/legoesm
./packages/core/legoesm
./packages/tools/legoesm
./packages/ocean/legoesm
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
    21	# lower bound decides it:
    22	#
    23	#   coloring_gap = n_rounds - max_degree
    24	#     > 0  -> the finite greedy-order search has NOT reached the graph's
    25	#             lower bound; a better COLOURING cuts rounds without touching
    26	#             ownership.  Cheap lever, do that first.
    27	#     == 0 -> colouring is exhausted at this ownership; only a partitioner
    28	#             that lowers max_degree can cut rounds.  Expensive lever.
    29	#
    30	# (a) NUMBER PRODUCED: n_rounds, max_degree and their gap per
    31	#     (method x n_dev) at the production working points.
    32	# (b) CONFIRMS a cheap fix: gap > 0 at any production (ppermute) row.
    33	#     REFUTES it: gap == 0 on every production row -> the 7-14 day
    34	#     ownership build is the only remaining path, and is then justified.
    35	# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
    36	#     step.  It cannot be shrunk further onto small meshes: measured
    37	#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
    38	#     but EVERY one of those rows auto-selects the ALLGATHER strategy
    39	#     (cells/device below the threshold), so production runs no ppermute
    40	#     schedule there and the number is counterfactual.  Only meshes big
    41	#     enough to keep cells/device above the threshold answer the question.
    42	#
    43	# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
    44	# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
    45	#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
    46	#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
    47	# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
    48	# A scan that misses the known answer is a broken instrument, and its s10
    49	# row must not be believed.
    50	#
    51	# lloyd=0 throughout: it is what the reference census used AND the cache key
    52	# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
    53	# synthetic scaling mesh, recorded in every row's metadata so it can never be
    54	# read back as a production SCVT receipt.
    55	#
    56	# Arms run cheapest-first and each writes its own JSON, so a later arm that
    57	# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
    58	# last and is the one genuinely at risk: the scorer is known not to have
    59	# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
    60	#
    61	# SUBMIT (from the repo root):
    62	#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
    63	# ===========================================================================
    64	set -uo pipefail
    65	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    66	export JAX_PLATFORMS=cpu
    67	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    68	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    69	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    70	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    71	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    72	
    73	OUT=results/a1/mpas_schedule_cost
    74	mkdir -p "$OUT"
    75	BENCH=scripts/bench/bench_voronoi_partition_methods.py
    76	
    77	# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
    78	# missing partitioner is reported as "unavailable" rather than substituted.
    79	# Say which python and whether metis is really there, so a two-method table
    80	# cannot be misread as a three-method one.
    81	echo "[scan] python=$PY"
    82	"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
    83	  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
    84	
    85	run_arm () {  # $1=level  $2=rank-counts  $3=label
    86	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
    87	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
    88	  "$PY" "$BENCH" \
    89	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
    90	      --methods geometric,sfc,metis --schedule-cost \
    91	      --out "$OUT/schedule_cost_s$1.json"
    92	  echo "[scan] arm $3 exit=$?"
    93	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
    94	}
    95	
    96	# Arms 1-2: KNOWN answers (the docstring census) — instrument validation.
    97	run_arm 8  64,128  "1/3 VALIDATION s8"
    98	run_arm 9  64,128  "2/3 VALIDATION s9"
    99	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   100	run_arm 10 128     "3/3 UNKNOWN s10"
   101	
   102	echo "SCAN_DONE"

exec
/usr/bin/bash -lc 'rg -n -C 3 "''^(def _build_ppermute_schedule|def spmd_schedule_cost|SPMD_HALO_DEPTH|def reorder_voronoi_for_sharding|reorder_target|n_rounds|max_degree)" packages/core/legoesm/parallel/sharded_dynamics.py packages/core/legoesm/parallel/voronoi_partition.py && nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '"'560,1100p' && nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '1,700p' && nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '1,390p' && nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '1,310p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 125ms:
packages/core/legoesm/parallel/sharded_dynamics.py-1244-#: consumed by both the production step factory and ``spmd_schedule_cost``:
packages/core/legoesm/parallel/sharded_dynamics.py-1245-#: a score computed at a different depth describes a different comm graph, and
packages/core/legoesm/parallel/sharded_dynamics.py-1246-#: two independently hardcoded 3s let production drift unnoticed.
packages/core/legoesm/parallel/sharded_dynamics.py:1247:SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py-1248-
packages/core/legoesm/parallel/sharded_dynamics.py-1249-
packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
packages/core/legoesm/parallel/sharded_dynamics.py-1251-                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py-1252-                       ppermute_cells_per_device_threshold=2_000):
packages/core/legoesm/parallel/sharded_dynamics.py-1253-    """How much halo communication one ownership choice costs, computed offline.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1759-    return best_colors, max_degree
packages/core/legoesm/parallel/sharded_dynamics.py-1760-
packages/core/legoesm/parallel/sharded_dynamics.py-1761-
packages/core/legoesm/parallel/sharded_dynamics.py:1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py-1763-                             edges_per, max_lc, max_le):
packages/core/legoesm/parallel/sharded_dynamics.py-1764-    """Build a ppermute-based halo exchange schedule.
packages/core/legoesm/parallel/sharded_dynamics.py-1765-
--
packages/core/legoesm/parallel/voronoi_partition.py-1024-    return resolve_partition_method(method)
packages/core/legoesm/parallel/voronoi_partition.py-1025-
packages/core/legoesm/parallel/voronoi_partition.py-1026-
packages/core/legoesm/parallel/voronoi_partition.py:1027:def reorder_voronoi_for_sharding(
packages/core/legoesm/parallel/voronoi_partition.py-1028-    mesh: VoronoiMesh,
packages/core/legoesm/parallel/voronoi_partition.py-1029-    n_devices: int,
packages/core/legoesm/parallel/voronoi_partition.py-1030-    *,
   560	        return key in self._cache
   561	
   562	    def clear_cache(self):
   563	        """Drop all cached executables (forces recompilation on next call)."""
   564	        self._cache.clear()
   565	        self._out_sharding_cache.clear()
   566	        self.compile_count = 0
   567	
   568	    @property
   569	    def cache_size(self) -> int:
   570	        """Number of distinct compiled executables currently cached."""
   571	        return len(self._cache)
   572	
   573	
   574	class _SingleDeviceStep:
   575	    """Thin JIT wrapper for single-device execution (no sharding).
   576	
   577	    This mirrors the :class:`CompiledShardedStep` API but uses a single
   578	    ``jax.jit`` without sharding constraints.  The ``dt`` argument is
   579	    traced, not static, so one compiled program handles all dt values.
   580	
   581	    When ``physics_fn`` is provided, it is captured in the jitted
   582	    closure (Python callables cannot be traced by JAX).  This means
   583	    each distinct ``physics_fn`` object identity produces a separate
   584	    cache entry, which is the correct behavior — different physics
   585	    functions produce different XLA programs.
   586	    """
   587	
   588	    def __init__(self, model):
   589	        self._model = model
   590	        self.compile_count = 0
   591	        self.last_compile_time_s = 0.0
   592	        # Cache key: (structure, leaf_meta, has_physics, physics_fn_id)
   593	        self._cache: dict[tuple, object] = {}
   594	
   595	    def _cache_key(self, state, physics_fn=None):
   596	        structure = jax.tree.structure(state)
   597	        leaves = jax.tree.leaves(state)
   598	        leaf_meta = tuple(
   599	            (l.shape, l.dtype) if hasattr(l, "shape") else (None, None)
   600	            for l in leaves
   601	        )
   602	        has_physics = physics_fn is not None
   603	        phys_id = id(physics_fn) if has_physics else None
   604	        return (structure, leaf_meta, has_physics, phys_id)
   605	
   606	    def _get_executable(self, state, physics_fn=None):
   607	        key = self._cache_key(state, physics_fn)
   608	        if key not in self._cache:
   609	            model = self._model
   610	            t0 = time.monotonic()
   611	            if physics_fn is not None and hasattr(model, "step_with_physics"):
   612	                _refuse_stateful_physics_unthreaded_wrapper(physics_fn)
   613	                # Capture physics_fn in the closure so JAX doesn't
   614	                # try to trace it as an array argument.
   615	                _phys = physics_fn
   616	
   617	                @jax.jit
   618	                def _step(s, dt):
   619	                    return model.step_with_physics(s, dt, _phys)
   620	            else:
   621	                @jax.jit
   622	                def _step(s, dt):
   623	                    return model.step(s, dt)
   624	            self._cache[key] = _step
   625	            self.compile_count += 1
   626	            self.last_compile_time_s = time.monotonic() - t0
   627	            logger.info(
   628	                "_SingleDeviceStep: compiled executable #%d (%.3fs)",
   629	                self.compile_count, self.last_compile_time_s,
   630	            )
   631	        return self._cache[key]
   632	
   633	    def __call__(self, state, dt, physics_fn=None):
   634	        fn = self._get_executable(state, physics_fn)
   635	        return fn(state, dt)
   636	
   637	    def is_compiled_for(self, state, physics_fn=None) -> bool:
   638	        key = self._cache_key(state, physics_fn)
   639	        return key in self._cache
   640	
   641	    def clear_cache(self):
   642	        self._cache.clear()
   643	        self.compile_count = 0
   644	
   645	    @property
   646	    def cache_size(self) -> int:
   647	        return len(self._cache)
   648	
   649	
   650	# ======================================================================
   651	# Sharded step construction
   652	# ======================================================================
   653	
   654	# Sub-face tile factors whose tiled step is bit-identity-validated against the
   655	# global op. kt=2 (24 devices) and kt=3 (54) were validated standalone AND in a
   656	# 2-node multi-controller run (rel=0.0). Anything else replicates the global
   657	# state instead of sharding it, so it is REFUSED rather than silently run
   658	# (#1360). Grow this set only together with the validation evidence.
   659	VALIDATED_TILE_FACTORS = frozenset({2, 3})
   660	
   661	
   662	def make_sharded_step(
   663	    model,
   664	    config: DeviceConfig,
   665	    halo_exchange_fn=None,
   666	    *,
   667	    n: int = 0,
   668	    nlev: int = 1,
   669	):
   670	    """Wrap a dynamics model's step function for multi-device execution.
   671	
   672	    Returns a callable (:class:`CompiledShardedStep` or
   673	    :class:`_SingleDeviceStep`) that:
   674	
   675	    1. On the first call, JIT-compiles the step function with explicit
   676	       output sharding constraints.
   677	    2. On subsequent calls with the same state structure, reuses the
   678	       cached compiled executable — no recompilation.
   679	    3. Returns the updated sharded state.
   680	
   681	    The returned callable has the signature::
   682	
   683	        step(state, dt, physics_fn=None) -> state
   684	
   685	    It also exposes ``compile_count``, ``is_compiled_for(state)``,
   686	    ``cache_size``, and ``clear_cache()`` for observability.
   687	
   688	    Parameters
   689	    ----------
   690	    model
   691	        A dynamics model with a ``.step(state, dt)`` method.  Typically
   692	        a ``PrimitiveEquationModel`` or ``ShallowWaterModel``.
   693	    config : DeviceConfig
   694	        Device configuration from :func:`create_device_mesh`.
   695	    halo_exchange_fn : callable, optional
   696	        Custom halo exchange function ``f(state) -> state`` applied
   697	        after each dynamics step.  If ``None``, the model's built-in
   698	        halo exchange (via ``pad_halo``) is used.
   699	    n : int, optional
   700	        Per-face resolution — logging/prewarm metadata forwarded to
   701	        ``activate_spmd_halo_backend``.  The old ppermute-vs-all_gather
   702	        volume auto-selection is RETIRED: ppermute is always selected;
   703	        all_gather only via explicit ``LEGOESM_SPMD_FORCE_ALLGATHER=1``.
   704	    nlev : int, optional
   705	        Number of vertical levels (logging only, see ``n``).
   706	
   707	    Returns
   708	    -------
   709	    CompiledShardedStep or _SingleDeviceStep
   710	        Callable with ``(state, dt, physics_fn=None) -> state``.
   711	
   712	    Notes
   713	    -----
   714	    For face-only sharding (1/2/3/6 devices) this function ACTIVATES
   715	    the explicit SPMD halo backend
   716	    (``cubesphere_exchange.activate_spmd_halo_backend``): ``pad_halo``
   717	    then routes through shard_map ppermute kernels (multiface; one-face
   718	    at halo=1 with 6 devices) instead of relying on XLA's implicit
   719	    cross-shard reads.  Letting GSPMD auto-insert collectives for the
   720	    cross-face reads — the pre-activation behavior this Notes section
   721	    used to describe — replicates ALL compute per device (HLO probe job
   722	    8456476); the all_gather kernels survive only as the explicit
   723	    ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` diagnostic.
   724	
   725	    For sub-face tiling (>6 devices), additional tile-boundary
   726	    exchange is needed; the SPMD halo backend is NOT activated there
   727	    (the tiled path is unvalidated — bench guards exclude it).
   728	    """
   729	    if config.mesh is None:
   730	        logger.info("make_sharded_step: single-device mode, using plain JIT")
   731	        return _SingleDeviceStep(model)
   732	
   733	    # Activate explicit SPMD halo exchange for face-sharded cubed-sphere.
   734	    # This replaces implicit cross-shard reads with explicit
   735	    # collective-permute rounds, producing much better XLA communication
   736	    # patterns.
   737	    #
   738	    # Iter-49 generalised activation from "exactly 6 devices" to "any
   739	    # divisor of 6" (1, 2, 3, 6) on the allgather kernels; the
   740	    # ppermute-multiface refit then made ppermute the DEFAULT exchange
   741	    # for every face-sharded count and at halo=2 — the allgather
   742	    # variant provably replicated ALL compute per device (HLO probe job
   743	    # 8456476, per-device FLOPs ratio 1.00 at 2 devices) and is now an
   744	    # explicit diagnostic opt-in only (LEGOESM_SPMD_FORCE_ALLGATHER=1).
   745	    _n = config.n_devices
   746	    _tiling = getattr(config, 'tiling', (1, 1))
   747	    _face_ok = (_n in (1, 2, 3, 6) and _tiling == (1, 1))
   748	    # 6*kt^2 sub-face tiling: the tiled ppermute EXCHANGE is serial-
   749	    # exact (h1+h2, offsets+raw, corners — probe job 8464648), but the
   750	    # DYCORE is not yet tile-aware: consumers slice padded arrays with
   751	    # full-face (n+2h) indexing (operators_cdgrid.py:555/638/766,
   752	    # operators_3d.py:85, fv_tp_2d.py:1024, fv3_sw_core.py:1358 —
   753	    # codex review) and staggered (n+1) leaves cannot shard over tile
   754	    # axes (IndivisibleError, probe job 8464703).  Activation is
   755	    # therefore EXPERIMENTAL and opt-in only; the P4 milestone
   756	    # (tile-aware consumers + staggered-leaf ownership layout) flips
   757	    # the default.
   758	    # UPDATE (2026-06-14): the tile-aware CONSUMERS now EXIST and are
   759	    # bit-identity-validated standalone — the full tiled production SW
   760	    # tendency ``make_tiled_fv3_sw_tendencies_stage_2d`` (momentum + mass-PPM,
   761	    # in-stage scalar/vector halos + deep-h pre-pad, staggered-leaf
   762	    # lower-owns-shared reassembly) passes np24 (kt=2) + np54 (kt=3)
   763	    # bit-identity vs the global op AND a 2-node multi-controller run
   764	    # (``scripts/validate/validate_tiled_fv3_sw_multinode.py``, rel=0.0).  What
   765	    # remains for the default-flip is WIRING that stage into THIS step (this
   766	    # function still calls the full-face operators); the 3D
   767	    # ``fv3_hydrostatic_tendencies`` tiling is in progress (dgrid_vorticity
   768	    # 4D-tiled).  See ``tiled_production_cdgrid.py`` +
   769	    # ``docs/performance/scaling/cube_production_tiling_design.md``.  NOT Ginsburg-benchable
   770	    # (np>6 anti-scales on Gloo-TCP/PCIe) — future-HW capability.
   771	    import os as _os
   772	    _tiled_requested = (
   773	        _os.environ.get("LEGOESM_TILED_SPMD", "0") == "1"
   774	        and _tiling[0] == _tiling[1] and _tiling[0] >= 2
   775	        and _n == 6 * _tiling[0] * _tiling[1]
   776	    )
   777	    # #1360: an UNVALIDATED kt used to fall through this branch silently, which
   778	    # left the step replicating the GLOBAL state on every device. The user then
   779	    # saw an opaque XLA argument-size error --
   780	    #   "The byte size of input/output arguments (83247045120) exceeds the base
   781	    #    limit (63820333056)"  (job 26495955, C768/L60 f32 at kt=4/96 devices)
   782	    # -- which reads as an OOM, not as "this tiling is not supported". That is
   783	    # the silent-fallback pattern the dispatch-hardening rule exists to kill:
   784	    # refuse loudly instead, naming what IS validated.
   785	    if _tiled_requested and _tiling[0] not in VALIDATED_TILE_FACTORS:
   786	        raise ValueError(
   787	            f"tiled cube SPMD is bit-identity-validated only at kt in "
   788	            f"{sorted(VALIDATED_TILE_FACTORS)} (6*kt^2 = "
   789	            f"{[6 * k * k for k in sorted(VALIDATED_TILE_FACTORS)]} devices); "
   790	            f"got kt={_tiling[0]} ({_n} devices). Running it would NOT shard: "
   791	            f"the step falls back to replicating the global state on every "
   792	            f"device and dies with an XLA argument-size error that looks like "
   793	            f"an OOM (#1360). Validate that kt the way kt=2/3 were "
   794	            f"(tiled-vs-global bit identity + a multi-controller run, "
   795	            f"scripts/validate/validate_tiled_fv3_sw_multinode.py) and add it "
   796	            f"to VALIDATED_TILE_FACTORS, or use a validated device count.")
   797	    _tiled_ok = _tiled_requested
   798	    if ((_face_ok or _tiled_ok)
   799	            and config.mesh is not None
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
   191	    n_local_vertices: int
   192	
   193	    # Global indices of local entities (owned first, then halo)
   194	    local_cells: np.ndarray       # (n_local_cells,)
   195	    local_edges: np.ndarray       # (n_local_edges,)
   196	    local_vertices: np.ndarray    # (n_local_vertices,)
   197	
   198	    # Global-to-local mapping (-1 for non-local entities)
   199	    cell_g2l: np.ndarray          # (nCells_global,)
   200	    edge_g2l: np.ndarray          # (nEdges_global,)
   201	    vertex_g2l: np.ndarray        # (nVertices_global,)
   202	
   203	    # Communication schedules
   204	    cell_comm: HaloCommSchedule
   205	    edge_comm: HaloCommSchedule
   206	    vertex_comm: HaloCommSchedule
   207	
   208	
   209	# ============================================================================
   210	# Partitioners
   211	# ============================================================================
   212	
   213	def partition_cells_geometric(mesh: VoronoiMesh, n_ranks: int) -> np.ndarray:
   214	    """Partition cells via Recursive Coordinate Bisection (RCB).
   215	
   216	    Uses cell-center Cartesian coordinates on the unit sphere.
   217	
   218	    Parameters
   219	    ----------
   220	    mesh : VoronoiMesh
   221	    n_ranks : int
   222	
   223	    Returns
   224	    -------
   225	    cell_owner : np.ndarray, shape (nCells,), dtype int32
   226	        ``cell_owner[c]`` is the rank that owns cell ``c``.
   227	    """
   228	    coords = np.stack([
   229	        np.asarray(mesh.xCell) / mesh.radius,
   230	        np.asarray(mesh.yCell) / mesh.radius,
   231	        np.asarray(mesh.zCell) / mesh.radius,
   232	    ], axis=1)
   233	    return _rcb(coords, n_ranks)
   234	
   235	
   236	def _rcb(coords: np.ndarray, n_ranks: int) -> np.ndarray:
   237	    """Recursive Coordinate Bisection on a point cloud."""
   238	    n = len(coords)
   239	    if n_ranks <= 1 or n <= 1:
   240	        return np.zeros(n, dtype=np.int32)
   241	
   242	    axis = int(np.argmax(np.ptp(coords, axis=0)))
   243	    order = np.argsort(coords[:, axis])
   244	
   245	    n_left_ranks = n_ranks // 2
   246	    n_right_ranks = n_ranks - n_left_ranks
   247	    split = max(1, min(n - 1, n * n_left_ranks // n_ranks))
   248	
   249	    left, right = order[:split], order[split:]
   250	    result = np.empty(n, dtype=np.int32)
   251	    result[left] = _rcb(coords[left], n_left_ranks)
   252	    result[right] = _rcb(coords[right], n_right_ranks) + n_left_ranks
   253	    return result
   254	
   255	
   256	def partition_cells_metis(mesh: VoronoiMesh, n_ranks: int) -> np.ndarray:
   257	    """Partition cells via METIS k-way graph partitioning.
   258	
   259	    Requires the ``pymetis`` package.
   260	
   261	    Parameters
   262	    ----------
   263	    mesh : VoronoiMesh
   264	    n_ranks : int
   265	
   266	    Returns
   267	    -------
   268	    cell_owner : np.ndarray, shape (nCells,), dtype int32
   269	    """
   270	    try:
   271	        import pymetis
   272	    except ImportError as exc:
   273	        raise ImportError(
   274	            "METIS partitioning requires pymetis.  "
   275	            "Install with: pip install pymetis"
   276	        ) from exc
   277	
   278	    coc = np.asarray(mesh.cellsOnCell)
   279	    nec = np.asarray(mesh.nEdgesOnCell)
   280	    adjacency = []
   281	    for c in range(mesh.nCells):
   282	        nbrs = [int(coc[k, c]) for k in range(int(nec[c])) if coc[k, c] >= 0]
   283	        adjacency.append(np.array(nbrs, dtype=np.int32))
   284	
   285	    _, membership = pymetis.part_graph(n_ranks, adjacency=adjacency)
   286	    return np.array(membership, dtype=np.int32)
   287	
   288	
   289	def _hilbert_xy2d(order: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
   290	    """Hilbert-curve distance ``d`` for integer grid coords ``(x, y)``.
   291	
   292	    Vectorized form of the canonical Wikipedia ``xy2d`` integer algorithm on a
   293	    ``2^order x 2^order`` grid (rotation uses the full side length ``n``, not the
   294	    current level ``s``).  Returns a bijection ``[0, n)^2 -> [0, n^2)`` whose
   295	    1-D ordering preserves 2-D locality: cells adjacent on the curve are spatially
   296	    close, which keeps each contiguous partition compact (small halo surface).
   297	    """
   298	    n = 1 << order
   299	    x = x.astype(np.int64).copy()
   300	    y = y.astype(np.int64).copy()
   301	    d = np.zeros(x.shape, dtype=np.int64)
   302	    s = n >> 1
   303	    while s > 0:
   304	        rx = ((x & s) > 0).astype(np.int64)
   305	        ry = ((y & s) > 0).astype(np.int64)
   306	        d += s * s * ((3 * rx) ^ ry)
   307	        # rot(n, x, y, rx, ry): reflect when ry==0 (and x,y when rx==1), then swap.
   308	        ry0 = ry == 0
   309	        flip = ry0 & (rx == 1)
   310	        x = np.where(flip, n - 1 - x, x)
   311	        y = np.where(flip, n - 1 - y, y)
   312	        tx = np.where(ry0, y, x)
   313	        ty = np.where(ry0, x, y)
   314	        x, y = tx, ty
   315	        s >>= 1
   316	    return d
   317	
   318	
   319	def hilbert_cell_keys(mesh: VoronoiMesh, order: int = _DEFAULT_HILBERT_ORDER) -> np.ndarray:
   320	    """Per-cell Hilbert space-filling-curve key from cell (lat, lon).
   321	
   322	    Maps each cell center to a ``2^order x 2^order`` (lon, lat) grid and returns
   323	    its Hilbert distance.  Sorting cells by this key yields a 1-D ordering with
   324	    strong 2-D spatial locality — used to build compact, contiguous partitions
   325	    and locality-friendly local indexings.
   326	
   327	    Parameters
   328	    ----------
   329	    mesh : VoronoiMesh
   330	    order : int
   331	        SFC grid resolution (side = ``2^order``).
   332	
   333	    Returns
   334	    -------
   335	    np.ndarray, shape (nCells,), dtype int64
   336	    """
   337	    two_pi = 2.0 * np.pi
   338	    lon = np.mod(np.asarray(mesh.lonCell, dtype=np.float64), two_pi)
   339	    lat = np.asarray(mesh.latCell, dtype=np.float64)
   340	    n = 1 << order
   341	    u = lon / two_pi                       # [0, 1)
   342	    v = (lat + 0.5 * np.pi) / np.pi        # [0, 1]
   343	    gx = np.clip((u * n).astype(np.int64), 0, n - 1)
   344	    gy = np.clip((v * n).astype(np.int64), 0, n - 1)
   345	    return _hilbert_xy2d(order, gx, gy)
   346	
   347	
   348	def partition_cells_sfc(
   349	    mesh: VoronoiMesh, n_ranks: int, order: int = _DEFAULT_HILBERT_ORDER,
   350	) -> np.ndarray:
   351	    """Partition cells into contiguous Hilbert space-filling-curve chunks.
   352	
   353	    Orders cells along a Hilbert curve, then assigns ``n_ranks`` balanced
   354	    contiguous runs.  Dependency-free (unlike METIS) and gives compact,
   355	    spatially-local partitions (smaller halos than RCB on irregular meshes).
   356	
   357	    Returns
   358	    -------
   359	    cell_owner : np.ndarray, shape (nCells,), dtype int32
   360	    """
   361	    n_cells = mesh.nCells
   362	    if n_ranks <= 1 or n_cells <= 1:
   363	        return np.zeros(n_cells, dtype=np.int32)
   364	    keys = hilbert_cell_keys(mesh, order)
   365	    order_idx = np.argsort(keys, kind="stable")
   366	    pos = np.empty(n_cells, dtype=np.int64)
   367	    pos[order_idx] = np.arange(n_cells, dtype=np.int64)
   368	    return (pos * n_ranks // n_cells).astype(np.int32)
   369	
   370	
   371	# ============================================================================
   372	# Halo computation
   373	# ============================================================================
   374	
   375	def compute_halo_cells(
   376	    cell_owner: np.ndarray,
   377	    cellsOnCell: np.ndarray,
   378	    maxEdges: int,
   379	    rank: int,
   380	    halo_depth: int,
   381	) -> set[int]:
   382	    """Compute the set of halo cells for *rank* up to *halo_depth* rings."""
   383	    owned = set(np.where(cell_owner == rank)[0].tolist())
   384	    halo: set[int] = set()
   385	    frontier = set(owned)
   386	    for _ in range(halo_depth):
   387	        new_frontier: set[int] = set()
   388	        for c in frontier:
   389	            for k in range(maxEdges):
   390	                nbr = int(cellsOnCell[k, c])
   391	                if nbr >= 0 and nbr not in owned and nbr not in halo:
   392	                    halo.add(nbr)
   393	                    new_frontier.add(nbr)
   394	        frontier = new_frontier
   395	    return halo
   396	
   397	
   398	# ============================================================================
   399	# Internal helpers
   400	# ============================================================================
   401	
   402	def _group_by_owner(halo_entities, owner_array):
   403	    """Group halo entities by their owner rank, sorted by global index."""
   404	    by_rank: dict[int, list[int]] = {}
   405	    for g in halo_entities:
   406	        r = int(owner_array[int(g)])
   407	        by_rank.setdefault(r, []).append(int(g))
   408	    for r in by_rank:
   409	        by_rank[r].sort()
   410	    return by_rank
   411	
   412	
   413	def _assemble_schedule(recv_by_rank, send_by_rank, g2l, neighbor_ranks):
   414	    """Build HaloCommSchedule from per-rank send/recv lists."""
   415	    send_idx_all: list[int] = []
   416	    recv_idx_all: list[int] = []
   417	    send_counts: list[int] = []
   418	    recv_counts: list[int] = []
   419	    active_ranks: list[int] = []
   420	
   421	    for r in neighbor_ranks:
   422	        s = send_by_rank.get(r, [])
   423	        rv = recv_by_rank.get(r, [])
   424	        if not s and not rv:
   425	            continue
   426	        active_ranks.append(r)
   427	        send_idx_all.extend(int(g2l[g]) for g in s)
   428	        send_counts.append(len(s))
   429	        recv_idx_all.extend(int(g2l[g]) for g in rv)
   430	        recv_counts.append(len(rv))
   431	
   432	    return HaloCommSchedule(
   433	        neighbor_ranks=tuple(active_ranks),
   434	        send_counts=tuple(send_counts),
   435	        recv_counts=tuple(recv_counts),
   436	        send_idx=(jnp.array(send_idx_all, dtype=jnp.int32)
   437	                  if send_idx_all
   438	                  else jnp.empty(0, dtype=jnp.int32)),
   439	        recv_idx=(jnp.array(recv_idx_all, dtype=jnp.int32)
   440	                  if recv_idx_all
   441	                  else jnp.empty(0, dtype=jnp.int32)),
   442	    )
   443	
   444	
   445	def build_batched_halo_schedule(
   446	    cell_comm: HaloCommSchedule,
   447	    edge_comm: HaloCommSchedule,
   448	) -> BatchedHaloSchedule:
   449	    """Join the cell and edge comm schedules into one union-neighbor schedule.
   450	
   451	    Per union neighbor (sorted union of the two neighbor lists), the
   452	    per-entity send/recv index slices are re-laid-out in union-neighbor
   453	    order; ranks absent from one entity schedule get a zero count for
   454	    that entity.  Everything here is host-side layout math (numpy) run
   455	    once at layout-build time — the resulting index arrays are JIT
   456	    constants.
   457	
   458	    The per-neighbor message sequence stays in sorted-rank order on every
   459	    rank, exactly like the entity schedules it replaces, so the blocking
   460	    ``sendrecv`` pairing properties of the existing exchange carry over
   461	    unchanged.
   462	    """
   463	
   464	    def _per_neighbor(comm: HaloCommSchedule):
   465	        """rank -> (send_rows, recv_rows) numpy slices for one entity."""
   466	        send_idx = np.asarray(comm.send_idx)
   467	        recv_idx = np.asarray(comm.recv_idx)
   468	        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
   469	        s_off = r_off = 0
   470	        for i, r in enumerate(comm.neighbor_ranks):
   471	            s_cnt = comm.send_counts[i]
   472	            r_cnt = comm.recv_counts[i]
   473	            out[r] = (
   474	                send_idx[s_off:s_off + s_cnt],
   475	                recv_idx[r_off:r_off + r_cnt],
   476	            )
   477	            s_off += s_cnt
   478	            r_off += r_cnt
   479	        return out
   480	
   481	    cell_by_rank = _per_neighbor(cell_comm)
   482	    edge_by_rank = _per_neighbor(edge_comm)
   483	    union = sorted(set(cell_by_rank) | set(edge_by_rank))
   484	
   485	    _empty = np.empty(0, dtype=np.int32)
   486	    cell_send_chunks, cell_recv_chunks = [], []
   487	    edge_send_chunks, edge_recv_chunks = [], []
   488	    cell_send_counts, cell_recv_counts = [], []
   489	    edge_send_counts, edge_recv_counts = [], []
   490	    for r in union:
   491	        c_s, c_r = cell_by_rank.get(r, (_empty, _empty))
   492	        e_s, e_r = edge_by_rank.get(r, (_empty, _empty))
   493	        cell_send_chunks.append(c_s)
   494	        cell_recv_chunks.append(c_r)
   495	        edge_send_chunks.append(e_s)
   496	        edge_recv_chunks.append(e_r)
   497	        cell_send_counts.append(int(len(c_s)))
   498	        cell_recv_counts.append(int(len(c_r)))
   499	        edge_send_counts.append(int(len(e_s)))
   500	        edge_recv_counts.append(int(len(e_r)))
   501	
   502	    def _cat(chunks):
   503	        if chunks:
   504	            flat = np.concatenate(chunks).astype(np.int32)
   505	        else:
   506	            flat = np.empty(0, dtype=np.int32)
   507	        return jnp.asarray(flat)
   508	
   509	    return BatchedHaloSchedule(
   510	        neighbor_ranks=tuple(int(r) for r in union),
   511	        cell_send_counts=tuple(cell_send_counts),
   512	        cell_recv_counts=tuple(cell_recv_counts),
   513	        edge_send_counts=tuple(edge_send_counts),
   514	        edge_recv_counts=tuple(edge_recv_counts),
   515	        cell_send_idx=_cat(cell_send_chunks),
   516	        cell_recv_idx=_cat(cell_recv_chunks),
   517	        edge_send_idx=_cat(edge_send_chunks),
   518	        edge_recv_idx=_cat(edge_recv_chunks),
   519	    )
   520	
   521	
   522	# ============================================================================
   523	# Main entry point
   524	# ============================================================================
   525	
   526	def partition_voronoi_mesh(
   527	    mesh: VoronoiMesh,
   528	    n_ranks: int,
   529	    rank: int,
   530	    *,
   531	    method: str = "auto",
   532	    halo_depth: int = 2,
   533	    cell_owner: np.ndarray | None = None,
   534	) -> VoronoiPartition:
   535	    """Partition a Voronoi mesh and build decomposition for *rank*.
   536	
   537	    Parameters
   538	    ----------
   539	    mesh : VoronoiMesh
   540	        Global mesh.
   541	    n_ranks : int
   542	        Total number of MPI ranks / devices.
   543	    rank : int
   544	        This rank (0-based).
   545	    method : str
   546	        ``"auto"`` (default: METIS if ``pymetis`` available, else RCB),
   547	        ``"geometric"`` (RCB), ``"metis"``, or ``"sfc"`` (Hilbert
   548	        space-filling-curve contiguous chunks).
   549	    halo_depth : int
   550	        Number of halo cell layers (default 2 for del4 support).
   551	    cell_owner : np.ndarray or None
   552	        Pre-computed cell ownership.  If ``None``, computed via *method*.
   553	
   554	    Returns
   555	    -------
   556	    VoronoiPartition
   557	    """
   558	    # Validate at entry on the static method value (CLAUDE.md: fail early) so an
   559	    # unknown method raises even when ``cell_owner`` is supplied or the method is
   560	    # otherwise unused.
   561	    method = resolve_partition_method(method)
   562	    if method not in ("geometric", "metis", "sfc"):
   563	        raise ValueError(f"Unknown partitioning method: {method!r}")
   564	    if cell_owner is None:
   565	        if method == "geometric":
   566	            cell_owner = partition_cells_geometric(mesh, n_ranks)
   567	        elif method == "metis":
   568	            cell_owner = partition_cells_metis(mesh, n_ranks)
   569	        else:  # "sfc" (validated above)
   570	            cell_owner = partition_cells_sfc(mesh, n_ranks)
   571	
   572	    # Convert mesh connectivity to numpy for the setup phase.
   573	    cellsOnCell = np.asarray(mesh.cellsOnCell)       # (maxEdges, nCells)
   574	    cellsOnEdge = np.asarray(mesh.cellsOnEdge)       # (2, nEdges)
   575	    cellsOnVertex = np.asarray(mesh.cellsOnVertex)   # (vDeg, nVertices)
   576	
   577	    # ------------------------------------------------------------------
   578	    # Cells
   579	    # ------------------------------------------------------------------
   580	    owned_cells = np.sort(np.where(cell_owner == rank)[0]).astype(np.int64)
   581	    halo_cells_set = compute_halo_cells(
   582	        cell_owner, cellsOnCell, mesh.maxEdges, rank, halo_depth,
   583	    )
   584	    halo_cells = np.array(sorted(halo_cells_set), dtype=np.int64)
   585	    local_cells = np.concatenate([owned_cells, halo_cells])
   586	    local_cells_set = set(owned_cells.tolist()) | halo_cells_set
   587	
   588	    cell_g2l = np.full(mesh.nCells, -1, dtype=np.int64)
   589	    for i, g in enumerate(local_cells):
   590	        cell_g2l[g] = i
   591	
   592	    # ------------------------------------------------------------------
   593	    # Edges
   594	    # ------------------------------------------------------------------
   595	    # An edge is local if at least one of its cells is local.
   596	    c1_all = np.asarray(cellsOnEdge[0])
   597	    c2_all = np.asarray(cellsOnEdge[1])
   598	    edge_local_mask = np.zeros(mesh.nEdges, dtype=bool)
   599	    for e in range(mesh.nEdges):
   600	        if int(c1_all[e]) in local_cells_set or int(c2_all[e]) in local_cells_set:
   601	            edge_local_mask[e] = True
   602	    all_local_edges = np.where(edge_local_mask)[0]
   603	
   604	    # Edge owner = owner of the cell with the smaller global index.
   605	    edge_owner_all = cell_owner[np.minimum(c1_all, c2_all)]
   606	
   607	    owned_edges = np.array(
   608	        sorted(int(e) for e in all_local_edges if edge_owner_all[e] == rank),
   609	        dtype=np.int64,
   610	    )
   611	    halo_edges = np.array(
   612	        sorted(int(e) for e in all_local_edges if edge_owner_all[e] != rank),
   613	        dtype=np.int64,
   614	    )
   615	    local_edges = np.concatenate([owned_edges, halo_edges])
   616	
   617	    edge_g2l = np.full(mesh.nEdges, -1, dtype=np.int64)
   618	    for i, g in enumerate(local_edges):
   619	        edge_g2l[g] = i
   620	
   621	    # ------------------------------------------------------------------
   622	    # Vertices
   623	    # ------------------------------------------------------------------
   624	    # A vertex is local if at least one of its cells is local.
   625	    cov_np = np.asarray(cellsOnVertex)
   626	    vertex_local_mask = np.zeros(mesh.nVertices, dtype=bool)
   627	    for v in range(mesh.nVertices):
   628	        for k in range(mesh.vertexDegree):
   629	            c = int(cov_np[k, v])
   630	            if c >= 0 and c in local_cells_set:
   631	                vertex_local_mask[v] = True
   632	                break
   633	    all_local_verts = np.where(vertex_local_mask)[0]
   634	
   635	    # Vertex owner = owner of the cell with the smallest global index
   636	    # among the vertex's cells.
   637	    cov_safe = np.where(cov_np >= 0, cov_np, mesh.nCells)
   638	    min_cell_v = np.min(cov_safe, axis=0)
   639	    vertex_owner_all = np.where(
   640	        min_cell_v < mesh.nCells,
   641	        cell_owner[np.minimum(min_cell_v, mesh.nCells - 1)],
   642	        0,
   643	    ).astype(np.int32)
   644	
   645	    owned_vertices = np.array(
   646	        sorted(int(v) for v in all_local_verts if vertex_owner_all[v] == rank),
   647	        dtype=np.int64,
   648	    )
   649	    halo_vertices = np.array(
   650	        sorted(int(v) for v in all_local_verts if vertex_owner_all[v] != rank),
   651	        dtype=np.int64,
   652	    )
   653	    local_vertices = np.concatenate([owned_vertices, halo_vertices])
   654	
   655	    vertex_g2l = np.full(mesh.nVertices, -1, dtype=np.int64)
   656	    for i, g in enumerate(local_vertices):
   657	        vertex_g2l[g] = i
   658	
   659	    # ------------------------------------------------------------------
   660	    # Communication schedules
   661	    # ------------------------------------------------------------------
   662	    # Recv side: group halo entities by their owner rank.
   663	    cell_recv = _group_by_owner(halo_cells, cell_owner)
   664	    edge_recv = _group_by_owner(halo_edges, edge_owner_all)
   665	    vertex_recv = _group_by_owner(halo_vertices, vertex_owner_all)
   666	
   667	    neighbor_ranks = sorted(cell_recv.keys())
   668	
   669	    # For each neighbor rank R, precompute which cells are in R's local
   670	    # domain (owned + halo).  This lets us determine which of our owned
   671	    # edges/vertices R needs as halo.
   672	    # Candidate ranks for the SEND schedule must be a SUPERSET of the cell-recv
   673	    # neighbours.  A rank can share only an EDGE or VERTEX boundary with me — it
   674	    # holds an edge/vertex I own in its halo — without its cell-halo reaching my
   675	    # cells, so it is absent from `neighbor_ranks` (= cell-recv owners) and the
   676	    # cell-neighbour-only `cell_to_nbr` would never mark it as needing that edge:
   677	    # I would not send, its blocking sendrecv to me would hang.  This is the
   678	    # np>=64 multi-node deadlock (asymmetric edge schedule; cells were fine).
   679	    # An edge/vertex spans exactly one cell-ring beyond the cell halo, so owners
   680	    # of cells within (halo_depth + 1) rings of my owned cells are a provably
   681	    # sufficient superset (an edge I own has one cell of mine and one neighbour
   682	    # cell; any rank needing it is within halo_depth of that neighbour cell,
   683	    # i.e. within halo_depth + 1 of my cell).
   684	    send_candidate_cells = compute_halo_cells(
   685	        cell_owner, cellsOnCell, mesh.maxEdges, rank, halo_depth + 1,
   686	    )
   687	    cell_to_nbr_ranks = sorted(
   688	        (set(neighbor_ranks)
   689	         | {int(cell_owner[c]) for c in send_candidate_cells})
   690	        - {rank}
   691	    )
   692	    cell_to_nbr: dict[int, set[int]] = {}
   693	    for R in cell_to_nbr_ranks:
   694	        R_owned = set(np.where(cell_owner == R)[0].tolist())
   695	        R_halo = compute_halo_cells(
   696	            cell_owner, cellsOnCell, mesh.maxEdges, R, halo_depth,
   697	        )
   698	        for c in R_owned | R_halo:
   699	            cell_to_nbr.setdefault(c, set()).add(R)
   700	
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
    33	     headroom is exactly ZERO.  Only a partitioner with a new objective
    34	     (minimize boundary max-degree, not edge cut) can lower the round
    35	     count.
    36	   * ``gap == 1`` -> INDISTINGUISHABLE from optimal here.  A Class 2 graph
    37	     genuinely needs ``Delta + 1``, and deciding Class 1 vs Class 2 is
    38	     NP-complete, so this does NOT establish that a better colouring
    39	     exists.  Ceiling either way: one round.
    40	   * ``gap >= 2`` -> at least ``gap - 1`` rounds of real recolouring
    41	     headroom (Vizing caps the optimum at ``Delta + 1``).
    42	
    43	   Consequence worth stating before any measurement: wherever the
    44	   multi-start search lands on ``Delta`` or ``Delta + 1`` — which it does
    45	   on every configuration probed so far — recolouring is capped at ONE
    46	   round out of 12-14, i.e. <= ~7%, and at zero where the gap is 0.  The
    47	   round count is an OWNERSHIP problem, not a colouring problem.
    48	
    49	   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
    50	   flag is the MPI lane's (default 2), while the schedule is scored at the
    51	   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
    52	   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
    53	   evaluations of the integrator actually run.  When
    54	   ``production_strategy`` is ``"allgather"`` (auto-selected below the
    55	   cells/device threshold) there is no ppermute schedule in production and
    56	   the round count is COUNTERFACTUAL; the row says so.
    57	2. **Step time** (optional pointer, NOT run here): drive the existing
    58	   MPI lane with ``bench_ocean_mpas_scaling.py --partition-method <m>``
    59	   (ocean) or ``bench_mpas_spmd_scaling.py --partition-method <m>``
    60	   (atmosphere SPMD) — one method per launch, same case otherwise
    61	   (controlled comparison).
    62	
    63	Guards: methods that are unavailable (``metis`` without ``pymetis``) are
    64	reported as ``"unavailable"`` — never silently substituted, so a table
    65	column can never claim METIS numbers that actually came from the RCB
    66	fallback.  Partition CORRECTNESS is asserted per row (every cell owned by
    67	exactly one rank; owner range valid) before any metric is recorded.
    68	
    69	Run:
    70	  python scripts/bench/bench_voronoi_partition_methods.py \
    71	      --subdivision 6 --rank-counts 2,4,8,16 --out results/partition_quality.json
    72	
    73	  # + the SPMD schedule depth (minutes to hours at subdiv>=8 — batch it):
    74	  python scripts/bench/bench_voronoi_partition_methods.py \
    75	      --subdivision 9 --rank-counts 64,128 --schedule-cost \
    76	      --out results/a1/schedule_cost_s9.json
    77	"""
    78	from __future__ import annotations
    79	
    80	import argparse
    81	import json
    82	import os
    83	import sys
    84	from pathlib import Path
    85	
    86	import numpy as np
    87	
    88	sys.path.insert(0, str(Path(__file__).resolve().parent))
    89	
    90	from metadata import annotate_incomplete, scaling_metadata  # noqa: E402
    91	
    92	METHODS = ("geometric", "sfc", "metis")
    93	
    94	
    95	def method_available(method: str) -> bool:
    96	    if method != "metis":
    97	        return True
    98	    try:
    99	        import pymetis  # noqa: F401
   100	
   101	        return True
   102	    except Exception:
   103	        return False
   104	
   105	
   106	def partition_quality(mesh, cell_owner: np.ndarray, n_ranks: int,
   107	                      halo_depth: int = 2) -> dict:
   108	    """Exact partition-quality metrics from a global owner array.
   109	
   110	    Serial enumeration of every rank (no MPI): the same halo construction
   111	    the runtime uses (``compute_halo_cells``), so the reported halo sizes
   112	    are the runtime's, not an estimate.
   113	    """
   114	    from legoesm.parallel.voronoi_partition import compute_halo_cells
   115	
   116	    n_cells = int(mesh.nCells)
   117	    # Correctness: every cell owned exactly once, owners in range, no
   118	    # empty rank (an empty rank would silently deflate the halo/owned
   119	    # ratio through the max(counts, 1) guard).
   120	    if n_cells == 0 or cell_owner.size == 0:
   121	        raise AssertionError("empty mesh / owner array")
   122	    if cell_owner.shape != (n_cells,):
   123	        raise AssertionError(f"owner shape {cell_owner.shape} != ({n_cells},)")
   124	    if cell_owner.min() < 0 or cell_owner.max() >= n_ranks:
   125	        raise AssertionError("owner out of range")
   126	    counts = np.bincount(cell_owner, minlength=n_ranks).astype(float)
   127	    if int(counts.sum()) != n_cells:
   128	        raise AssertionError("ownership does not cover the mesh")
   129	    if counts.min() <= 0:
   130	        raise AssertionError(
   131	            f"empty rank in partition (counts.min()={counts.min():.0f}) — "
   132	            f"a skipped rank corrupts every per-rank metric")
   133	
   134	    # Edge cut: edges whose two cells have different owners.
   135	    c1, c2 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
   136	    valid = (c1 >= 0) & (c2 >= 0)
   137	    edge_cut = int((cell_owner[c1[valid]] != cell_owner[c2[valid]]).sum())
   138	
   139	    halo_sizes = []
   140	    neighbor_counts = []
   141	    cells_on_cell = np.asarray(mesh.cellsOnCell)
   142	    max_edges = int(cells_on_cell.shape[0]) if cells_on_cell.ndim == 2 else 0
   143	    for r in range(n_ranks):
   144	        halo = compute_halo_cells(
   145	            cell_owner, mesh.cellsOnCell, mesh.maxEdges, r, halo_depth)
   146	        halo_sizes.append(len(halo))
   147	        neighbor_counts.append(
   148	            len(set(int(cell_owner[c]) for c in halo) - {r}))
   149	    _ = max_edges
   150	    halo_sizes = np.array(halo_sizes, dtype=float)
   151	    return {
   152	        "cells_per_rank_min": int(counts.min()),
   153	        "cells_per_rank_max": int(counts.max()),
   154	        "load_imbalance_max_over_mean": float(counts.max() / counts.mean()),
   155	        "edge_cut": edge_cut,
   156	        "edge_cut_fraction": float(edge_cut / max(int(valid.sum()), 1)),
   157	        "halo_cells_max": int(halo_sizes.max()),
   158	        "halo_cells_mean": float(halo_sizes.mean()),
   159	        "halo_owned_ratio_max": float(
   160	            (halo_sizes / np.maximum(counts, 1.0)).max()),
   161	        "neighbor_ranks_max": int(max(neighbor_counts)),
   162	    }
   163	
   164	
   165	def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
   166	    from legoesm.parallel.voronoi_partition import (
   167	        partition_cells_geometric,
   168	        partition_cells_metis,
   169	        partition_cells_sfc,
   170	    )
   171	
   172	    if method == "geometric":
   173	        return np.asarray(partition_cells_geometric(mesh, n_ranks))
   174	    if method == "sfc":
   175	        return np.asarray(partition_cells_sfc(mesh, n_ranks))
   176	    if method == "metis":
   177	        return np.asarray(partition_cells_metis(mesh, n_ranks))
   178	    raise ValueError(f"unknown partition method {method!r}; "
   179	                     f"expected one of {METHODS}")
   180	
   181	
   182	def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
   183	    """SPMD halo-schedule depth for one (method, n_ranks) candidate.
   184	
   185	    Thin wrapper over the production
   186	    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
   187	    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
   188	    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
   189	    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
   190	    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
   191	    lane's (2), and scoring the SPMD schedule at 2 would colour a different
   192	    graph.
   193	
   194	    Adds ``coloring_gap = n_rounds - max_degree`` (see the module docstring:
   195	    ``>0`` means recolouring still has room, ``0`` means only a new ownership
   196	    objective can help).
   197	
   198	    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
   199	    different reorder target, which would mis-slice the owned blocks — is
   200	    unreachable from here: this passes the raw mesh with the scorer's default
   201	    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
   202	    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
   203	    the call would therefore only swallow *unforeseen* failures into a row
   204	    that reads like an orderly skip, which is how a missing number turns into
   205	    a silently wrong table.  Rows already print as the sweep goes, so a raise
   206	    keeps the completed rungs in the log.
   207	    """
   208	    import time
   209	
   210	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   211	
   212	    t0 = time.perf_counter()
   213	    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
   214	    gap = int(cost["n_rounds"]) - int(cost["max_degree"])
   215	    return {
   216	        "n_rounds": int(cost["n_rounds"]),
   217	        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
   218	        "max_degree": int(cost["max_degree"]),
   219	        # The decisive column: which fix is worth costing at all.
   220	        "coloring_gap": gap,
   221	        "coloring_headroom": "recolor" if gap > 0 else "ownership_only",
   222	        "coloring_method": cost["coloring_method"],
   223	        "resolved_method": cost["resolved_method"],
   224	        "schedule_halo_depth": int(cost["halo_depth"]),
   225	        "cells_per_device": int(cost["cells_per_device"]),
   226	        # "allgather" => production runs no ppermute schedule here, so the
   227	        # round count above is COUNTERFACTUAL, not a cost production pays.
   228	        "production_strategy": cost["production_strategy"],
   229	        "score_seconds": round(time.perf_counter() - t0, 2),
   230	    }
   231	
   232	
   233	def main() -> int:
   234	    p = argparse.ArgumentParser(
   235	        description=__doc__,
   236	        formatter_class=argparse.RawDescriptionHelpFormatter)
   237	    p.add_argument("--subdivision", type=int, default=5,
   238	                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
   239	    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
   240	    p.add_argument("--halo-depth", type=int, default=2,
   241	                   help="Halo layers (runtime default 2, del4 support).")
   242	    p.add_argument("--methods", type=str, default=",".join(METHODS))
   243	    p.add_argument("--schedule-cost", action="store_true",
   244	                   help="Also score the SPMD ppermute halo-schedule depth "
   245	                        "(n_rounds vs max_degree) per method x rank count. "
   246	                        "Uses the production SPMD halo depth, NOT "
   247	                        "--halo-depth. Expensive: minutes per candidate at "
   248	                        "subdiv>=8 — run it under batch.")
   249	    p.add_argument("--lloyd", type=int, default=50,
   250	                   help="Lloyd relaxation iterations for the mesh. 50 = the "
   251	                        "production SCVT key; 0 = the LABELLED synthetic "
   252	                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
   253	                        "masquerade as a production receipt, and it must "
   254	                        "match the prewarmed cache key at subdiv>=9.")
   255	    p.add_argument("--out", type=str,
   256	                   default="results/a1/voronoi_partition_quality.json")
   257	    args = p.parse_args()
   258	
   259	    rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
   260	    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
   261	    for m in methods:
   262	        if m not in METHODS:
   263	            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
   264	    if not rank_counts or any(n < 2 for n in rank_counts):
   265	        raise SystemExit("--rank-counts needs integers >= 2")
   266	
   267	
   268	    from legoesm.grids.voronoi import create_voronoi_mesh
   269	    from legoesm.parallel.voronoi_partition import resolve_partition_method
   270	
   271	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
   272	                               lloyd_iterations=args.lloyd)
   273	    if max(rank_counts) > int(mesh.nCells):
   274	        raise SystemExit(
   275	            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
   276	            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
   277	    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
   278	          f"{int(mesh.nEdges)} edges; auto -> "
   279	          f"{resolve_partition_method('auto')!r}")
   280	
   281	    rows = []
   282	    for method in methods:
   283	        if not method_available(method):
   284	            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
   285	                  f"column omitted, never substituted")
   286	            rows.append({"method": method, "available": False})
   287	            continue
   288	        for n_ranks in rank_counts:
   289	            q = partition_quality(
   290	                mesh, owner_for(mesh, method, n_ranks), n_ranks,
   291	                halo_depth=args.halo_depth)
   292	            row = {"method": method, "available": True,
   293	                   "n_ranks": n_ranks, **q}
   294	            rows.append(row)
   295	            print(f"  {method:9s} np={n_ranks:3d} | "
   296	                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
   297	                  f"edge_cut={q['edge_cut']:6d} "
   298	                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
   299	                  f"halo max={q['halo_cells_max']:5d} "
   300	                  f"mean={q['halo_cells_mean']:8.1f} | "
   301	                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
   302	                  f"nbrs max={q['neighbor_ranks_max']}")
   303	            if args.schedule_cost:
   304	                sc = schedule_cost_row(mesh, method, n_ranks)
   305	                row["schedule"] = sc
   306	                note = ("  [COUNTERFACTUAL: production auto-selects "
   307	                        "allgather here, no ppermute schedule]"
   308	                        if sc["production_strategy"] == "allgather" else "")
   309	                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
   310	                      f"rounds={sc['n_rounds']:3d} "
   311	                      f"max_degree={sc['max_degree']:3d} "
   312	                      f"gap={sc['coloring_gap']:+d} "
   313	                      f"-> {sc['coloring_headroom']} "
   314	                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
   315	
   316	    payload = {
   317	        "rows": rows,
   318	        "auto_resolves_to": resolve_partition_method("auto"),
   319	        "step_time_pointer": (
   320	            "step-time per method: bench_ocean_mpas_scaling.py / "
   321	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   322	            "per launch, same case otherwise)"),
   323	        "metadata": annotate_incomplete(scaling_metadata(
   324	            grid="voronoi",
   325	            component="partitioning",
   326	            resolution=f"L{args.subdivision}",
   327	            n_levels=0,
   328	            precision="n/a",
   329	            decomposition="cell_partition",
   330	            solver_variant="n/a",
   331	            scaling_kind="partition-quality",
   332	            transport="none",
   333	            extra={"rank_counts": rank_counts, "methods": methods,
   334	                   "halo_depth": args.halo_depth,
   335	                   "schedule_cost": bool(args.schedule_cost),
   336	                   "lloyd_iterations": args.lloyd},
   337	        )),
   338	    }
   339	    outdir = os.path.dirname(args.out)
   340	    if outdir:
   341	        os.makedirs(outdir, exist_ok=True)
   342	    with open(args.out, "w") as f:
   343	        json.dump(payload, f, indent=2)
   344	    print(f"JSON: {args.out}")
   345	    return 0
   346	
   347	
   348	if __name__ == "__main__":
   349	    raise SystemExit(main())
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
   170	    "n_rounds, max_degree, gap, headroom",
   171	    [(14, 10, 4, "recolor"), (12, 12, 0, "ownership_only")])
   172	def test_coloring_gap_is_derived_not_assumed(
   173	        monkeypatch, n_rounds, max_degree, gap, headroom):
   174	    """The gap/headroom must be COMPUTED from the scorer's two fields.
   175	
   176	    Non-vacuity, the hard way: on every mesh small enough to test quickly the
   177	    real gap is 0 (measured L2/L3/L4 x geometric/sfc x nd 2-16 — the colourer
   178	    lands exactly on ``max_degree`` every time), so a real-mesh assertion
   179	    cannot tell a correct subtraction from a hardcoded ``0``; that exact
   180	    mutation passed the first version of this test.  Stubbing the production
   181	    scorer with a KNOWN non-zero gap is what makes the assertion able to
   182	    fail, and it also pins the branch that decides whether recolouring is
   183	    worth costing at all.
   184	    """
   185	    stub = {
   186	        "n_rounds": n_rounds, "max_degree": max_degree,
   187	        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
   188	        "resolved_method": "geometric", "halo_depth": 3,
   189	        "cells_per_device": 99_999, "production_strategy": "ppermute",
   190	    }
   191	    import legoesm.parallel.sharded_dynamics as sd
   192	    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
   193	
   194	    sc = mod.schedule_cost_row(object(), "geometric", 8)
   195	    assert sc["coloring_gap"] == gap
   196	    assert sc["coloring_headroom"] == headroom
   197	
   198	
   199	def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
   200	    """The schedule is scored at the SPMD production halo depth, NOT this
   201	    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
   202	    a different graph and quietly report the wrong lane's cost."""
   203	    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
   204	
   205	    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
   206	    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
   207	
   208	
   209	def test_schedule_cost_matches_the_production_scorer_exactly():
   210	    """Lock: the wrapper reports what the production scorer returns — it is
   211	    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
   212	    reports 8 rounds where the real depth-3 graph reports 12-14)."""
   213	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   214	
   215	    mesh = _mesh()
   216	    ref = spmd_schedule_cost(mesh, 2, method="sfc")
   217	    sc = mod.schedule_cost_row(mesh, "sfc", 2)
   218	    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
   219	                "coloring_method", "resolved_method", "cells_per_device",
   220	                "production_strategy"):
   221	        assert sc[key] == ref[key], key
   222	
   223	
   224	def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
   225	        tmp_path, monkeypatch):
   226	    """Off by default (it is the expensive layer); on, every scored row
   227	    carries the schedule block and the run records that it ran."""
   228	    out = tmp_path / "off.json"
   229	    monkeypatch.setattr(sys, "argv", [
   230	        "bench", "--subdivision", "2", "--rank-counts", "2",
   231	        "--methods", "geometric", "--out", str(out)])
   232	    assert mod.main() == 0
   233	    payload = json.loads(out.read_text())
   234	    assert "schedule" not in payload["rows"][0]
   235	    assert payload["metadata"]["extra"]["schedule_cost"] is False
   236	
   237	    out2 = tmp_path / "on.json"
   238	    monkeypatch.setattr(sys, "argv", [
   239	        "bench", "--subdivision", "2", "--rank-counts", "2",
   240	        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
   241	    assert mod.main() == 0
   242	    payload2 = json.loads(out2.read_text())
   243	    row = payload2["rows"][0]
   244	    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
   245	    assert payload2["metadata"]["extra"]["schedule_cost"] is True
   246	    # Mesh provenance: a lloyd=0 synthetic mesh must never be readable as a
   247	    # production SCVT receipt.
   248	    assert payload2["metadata"]["extra"]["lloyd_iterations"] == 50

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1210,1435p'; nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1610,1925p'; nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '990,1225p'; nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '1225,1490p'; nl -ba scripts/cluster/scaling_levante/_env.sh | sed -n '1,260p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 138ms:
  1210	    halo_cells_set : set[int]
  1211	        Initial halo (cellsOnCell ring + iter-23 owned-edge other-cells).
  1212	    cellsOnEdge_np : np.ndarray, shape (2, nEdges), int
  1213	        Global cellsOnEdge connectivity.
  1214	    n_passes : int
  1215	        Number of augmentation iterations.  2 is the minimum that
  1216	        closes the dycore's depth-2 operator chain; use a higher
  1217	        value only if a future operator extends the chain depth.
  1218	    """
  1219	    for _ in range(n_passes):
  1220	        cur_local_arr = np.concatenate([
  1221	            owned_cells_arr,
  1222	            np.fromiter(
  1223	                halo_cells_set,
  1224	                dtype=np.int64,
  1225	                count=len(halo_cells_set),
  1226	            ),
  1227	        ])
  1228	        # Edges where AT LEAST one ``cellsOnEdge`` is currently local.
  1229	        edge_one_in = (
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
  1421	        Each leaf has shape ``(n_dev, max_local_*)``.
  1422	    gather_cells : jnp.ndarray, (n_dev, max_local_cells)
  1423	    gather_edges : jnp.ndarray, (n_dev, max_local_edges)
  1424	    n_owned_cells : list[int]
  1425	    n_owned_edges : list[int]
  1426	    max_lc : int
  1427	    max_le : int
  1428	    partitions : list[VoronoiPartition]
  1429	        Per-device partition descriptors (for ppermute schedule building).
  1430	    cell_owner : np.ndarray, (nCells,)
  1431	        Cell ownership array.
  1432	    """
  1433	    import numpy as np
  1434	    from legoesm.parallel.voronoi_partition import (
  1435	        HaloCommSchedule,
  1610	            vertex_g2l=vertex_g2l,
  1611	            cell_comm=_dummy_comm,
  1612	            edge_comm=_dummy_comm,
  1613	            vertex_comm=_dummy_comm,
  1614	        )
  1615	        lm = build_local_mesh(gm_np, part)
  1616	        partitions.append(part)
  1617	        local_meshes_raw.append(lm)
  1618	
  1619	    # Uniform padding to the maximum local sizes across all devices.
  1620	    max_lc = max(p.n_local_cells for p in partitions)
  1621	    max_le = max(p.n_local_edges for p in partitions)
  1622	    max_lv = max(p.n_local_vertices for p in partitions)
  1623	
  1624	    local_meshes = [
  1625	        _pad_local_mesh_to(lm, max_lc, max_le, max_lv)
  1626	        for lm in local_meshes_raw
  1627	    ]
  1628	
  1629	    # Stack into a single pytree with a leading device dimension.
  1630	    stacked_meshes = jax.tree.map(
  1631	        lambda *leaves: jnp.stack(leaves, axis=0),
  1632	        *local_meshes,
  1633	    )
  1634	
  1635	    # Gather indices: for each device, global cell/edge indices of its
  1636	    # local entities (owned + halo), padded with 0 for ghost slots.
  1637	    gather_cells = np.zeros((n_dev, max_lc), dtype=np.int64)
  1638	    gather_edges = np.zeros((n_dev, max_le), dtype=np.int64)
  1639	    n_owned_cells = []
  1640	    n_owned_edges = []
  1641	    for rank, part in enumerate(partitions):
  1642	        gather_cells[rank, : part.n_local_cells] = part.local_cells
  1643	        gather_edges[rank, : part.n_local_edges] = part.local_edges
  1644	        n_owned_cells.append(part.n_owned_cells)
  1645	        n_owned_edges.append(part.n_owned_edges)
  1646	
  1647	    return (
  1648	        stacked_meshes,
  1649	        jnp.array(gather_cells),
  1650	        jnp.array(gather_edges),
  1651	        n_owned_cells,
  1652	        n_owned_edges,
  1653	        max_lc,
  1654	        max_le,
  1655	        partitions,
  1656	        cell_owner,
  1657	    )
  1658	
  1659	
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
   990	        dcEdge=pad_1d(mesh.dcEdge, pad_edges, fill=1.0),  # avoid /0
   991	        dvEdge=pad_1d(mesh.dvEdge, pad_edges, fill=0.0),  # zero flux
   992	        angleEdge=pad_1d(mesh.angleEdge, pad_edges, fill=0.0),
   993	        # --- weights and signs ---
   994	        weightsOnEdge=pad_2d_col(mesh.weightsOnEdge, pad_edges, fill=0.0),
   995	        kiteAreasOnVertex=mesh.kiteAreasOnVertex,  # vertex-indexed
   996	        fEdge=pad_1d(mesh.fEdge, pad_edges, fill=0.0),
   997	        fVertex=mesh.fVertex,  # vertex-indexed, unchanged
   998	        edgeSignOnCell=pad_2d_col(mesh.edgeSignOnCell, pad_cells, fill=0.0),
   999	        edgeSignOnVertex=mesh.edgeSignOnVertex,  # vertex-indexed
  1000	        meshDensity=pad_1d(mesh.meshDensity, pad_cells, fill=0.0),
  1001	        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
  1002	                             else pad_1d(mesh.subgrid_topo_stddev,
  1003	                                         pad_cells, fill=0.0)),
  1004	        land_frac=(None if mesh.land_frac is None
  1005	                   else pad_1d(mesh.land_frac, pad_cells, fill=0.0)),
  1006	    )
  1007	
  1008	
  1009	# ============================================================================
  1010	# Mesh reordering for JAX SPMD sharding
  1011	# ============================================================================
  1012	
  1013	def resolve_sharding_partition_method(method: str) -> str:
  1014	    """Concrete ownership for the SPMD/ppermute path (``auto`` -> ``sfc``).
  1015	
  1016	    Separate from :func:`resolve_partition_method`, whose ``auto`` prefers
  1017	    METIS: METIS minimizes edge CUT, while this path is bound by the number of
  1018	    sequential halo exchanges, and measured they move oppositely (subdiv-8 at
  1019	    128 devices: sfc 14 rounds, metis 19).  ONE definition, so a scorer that
  1020	    reports which ownership ran cannot drift from what the reorder does.
  1021	    """
  1022	    if method == "auto":
  1023	        return "sfc"
  1024	    return resolve_partition_method(method)
  1025	
  1026	
  1027	def reorder_voronoi_for_sharding(
  1028	    mesh: VoronoiMesh,
  1029	    n_devices: int,
  1030	    *,
  1031	    method: str = "auto",
  1032	) -> VoronoiMesh:
  1033	    """Reorder a Voronoi mesh so that JAX NamedSharding gives spatial locality.
  1034	
  1035	    Partitions cells via RCB (or METIS), then reorders cells, edges, and
  1036	    vertices so that entities owned by device 0 come first, then device 1,
  1037	    etc.  When JAX splits the reordered arrays into ``n_devices`` contiguous
  1038	    chunks along axis 0, each chunk corresponds to a spatially contiguous
  1039	    domain — minimizing cross-device communication in TRiSK stencils.
  1040	
  1041	    Parameters
  1042	    ----------
  1043	    mesh : VoronoiMesh
  1044	        Original global mesh.
  1045	    n_devices : int
  1046	        Number of devices (partitions).
  1047	    method : str
  1048	        ``"auto"`` -> ``"sfc"`` on THIS path (see below), ``"geometric"``
  1049	        (RCB), ``"metis"``, or ``"sfc"`` (Hilbert space-filling-curve
  1050	        contiguous chunks).
  1051	
  1052	    Returns
  1053	    -------
  1054	    VoronoiMesh
  1055	        Mesh with reordered entities and remapped connectivity.
  1056	    """
  1057	    # ``auto`` resolves to SFC HERE, not to the global METIS-if-available
  1058	    # policy.  This is the SPMD/ppermute path, where the cost that binds at
  1059	    # high device counts is the number of collective-permute ROUNDS -- equal
  1060	    # to the max degree of the post-reorder depth-3-plus-closure comm graph,
  1061	    # since the edge coloring already reaches that lower bound.  METIS
  1062	    # minimizes its ``cellsOnCell`` EDGE CUT, which is a different objective,
  1063	    # and measured on the real halo-aware layout the two move OPPOSITELY:
  1064	    #
  1065	    #   rounds (= max_degree)      64 dev   128 dev
  1066	    #     subdiv-8  geometric        16       21
  1067	    #     subdiv-8  sfc              12       14
  1068	    #     subdiv-8  metis            13       19
  1069	    #     subdiv-9  geometric        14       18
  1070	    #     subdiv-9  sfc              11       13
  1071	    #     subdiv-9  metis            14       18
  1072	    #
  1073	    # SFC wins at every mesh and device count; at SSP-RK3's 3 halo fills per
  1074	    # step, auto->metis would cost +15 collective-permutes/step at 128 on
  1075	    # both meshes.  This became live rather than theoretical when pymetis
  1076	    # became importable in the venvs, which silently flipped auto to the
  1077	    # worst choice for this path.  The high-count launchers pin ``sfc``
  1078	    # explicitly, so their receipts are unaffected either way.
  1079	    #
  1080	    # SCOPE: only this function.  ``initialize_voronoi_mpi`` (route-A MPI)
  1081	    # and ``partition_voronoi_mesh`` keep the global policy -- their halo
  1082	    # exchange is not this ppermute schedule and no census was run for them.
  1083	    # Validate at entry (CLAUDE.md: fail early) BEFORE the single-device shortcut,
  1084	    # so an unknown method raises even when no partitioning happens.
  1085	    method = resolve_sharding_partition_method(method)
  1086	    method = resolve_partition_method(method)
  1087	    if method not in ("geometric", "metis", "sfc"):
  1088	        raise ValueError(f"Unknown partitioning method: {method!r}")
  1089	    if n_devices <= 1:
  1090	        return mesh
  1091	
  1092	    # --- Partition cells ---
  1093	    if method == "geometric":
  1094	        cell_owner = partition_cells_geometric(mesh, n_devices)
  1095	    elif method == "metis":
  1096	        cell_owner = partition_cells_metis(mesh, n_devices)
  1097	    else:  # "sfc" (validated above)
  1098	        cell_owner = partition_cells_sfc(mesh, n_devices)
  1099	
  1100	    # --- Cell permutation: group by owner (primary), then order WITHIN each
  1101	    # owner by the Hilbert space-filling curve (secondary) so each contiguous
  1102	    # NamedSharding shard is spatially compact -> better cache/GPU locality and
  1103	    # smaller cross-shard stencil reach.  Ownership is unchanged; this sets only
  1104	    # the intra-shard order (the prior stable sort left it as arbitrary mesh
  1105	    # order).
  1106	    hkeys = hilbert_cell_keys(mesh)
  1107	    cell_perm = np.lexsort((hkeys, cell_owner))
  1108	    cell_inv = np.empty_like(cell_perm)
  1109	    cell_inv[cell_perm] = np.arange(len(cell_perm))
  1110	
  1111	    # --- Edge owner: owner of the cell with the smaller global index ---
  1112	    cellsOnEdge_np = np.asarray(mesh.cellsOnEdge)  # (2, nEdges)
  1113	    c0 = cellsOnEdge_np[0]
  1114	    c1 = cellsOnEdge_np[1]
  1115	    edge_owner = cell_owner[np.minimum(c0, c1)]
  1116	    edge_perm = np.argsort(edge_owner, kind="stable")
  1117	    edge_inv = np.empty_like(edge_perm)
  1118	    edge_inv[edge_perm] = np.arange(len(edge_perm))
  1119	
  1120	    # --- Vertex owner: owner of the cell with the smallest global index ---
  1121	    cellsOnVertex_np = np.asarray(mesh.cellsOnVertex)  # (vertexDegree, nVertices)
  1122	    cov_safe = np.where(cellsOnVertex_np >= 0, cellsOnVertex_np, mesh.nCells)
  1123	    min_cell_v = np.min(cov_safe, axis=0)
  1124	    vertex_owner = np.where(
  1125	        min_cell_v < mesh.nCells,
  1126	        cell_owner[np.minimum(min_cell_v, mesh.nCells - 1)],
  1127	        0,
  1128	    ).astype(np.int32)
  1129	    vert_perm = np.argsort(vertex_owner, kind="stable")
  1130	    vert_inv = np.empty_like(vert_perm)
  1131	    vert_inv[vert_perm] = np.arange(len(vert_perm))
  1132	
  1133	    # --- Helper: remap connectivity values through an inverse permutation ---
  1134	    def remap_conn(conn, inv_perm):
  1135	        """Remap integer connectivity array: old_global → new_global."""
  1136	        arr = np.asarray(conn)
  1137	        valid = arr >= 0
  1138	        safe = np.where(valid, arr, 0)
  1139	        remapped = np.where(valid, inv_perm[safe], -1)
  1140	        return jnp.array(remapped, dtype=conn.dtype)
  1141	
  1142	    # --- Helper: reorder along entity axis (last axis for (K, nEntities)) ---
  1143	    def reorder_col(arr, perm):
  1144	        """Reorder columns: arr[:, perm] for 2D, arr[perm] for 1D."""
  1145	        a = np.asarray(arr)
  1146	        if a.ndim == 1:
  1147	            return jnp.array(a[perm], dtype=arr.dtype)
  1148	        return jnp.array(a[:, perm], dtype=arr.dtype)
  1149	
  1150	    def reorder_1d(arr, perm):
  1151	        a = np.asarray(arr)
  1152	        return jnp.array(a[perm], dtype=arr.dtype)
  1153	
  1154	    reordered = VoronoiMesh(
  1155	        nCells=mesh.nCells,
  1156	        nEdges=mesh.nEdges,
  1157	        nVertices=mesh.nVertices,
  1158	        maxEdges=mesh.maxEdges,
  1159	        vertexDegree=mesh.vertexDegree,
  1160	        radius=mesh.radius,
  1161	        # Cell coordinates (reorder by cell_perm)
  1162	        latCell=reorder_1d(mesh.latCell, cell_perm),
  1163	        lonCell=reorder_1d(mesh.lonCell, cell_perm),
  1164	        xCell=reorder_1d(mesh.xCell, cell_perm),
  1165	        yCell=reorder_1d(mesh.yCell, cell_perm),
  1166	        zCell=reorder_1d(mesh.zCell, cell_perm),
  1167	        # Edge coordinates (reorder by edge_perm)
  1168	        latEdge=reorder_1d(mesh.latEdge, edge_perm),
  1169	        lonEdge=reorder_1d(mesh.lonEdge, edge_perm),
  1170	        xEdge=reorder_1d(mesh.xEdge, edge_perm),
  1171	        yEdge=reorder_1d(mesh.yEdge, edge_perm),
  1172	        zEdge=reorder_1d(mesh.zEdge, edge_perm),
  1173	        # Vertex coordinates (reorder by vert_perm)
  1174	        latVertex=reorder_1d(mesh.latVertex, vert_perm),
  1175	        lonVertex=reorder_1d(mesh.lonVertex, vert_perm),
  1176	        xVertex=reorder_1d(mesh.xVertex, vert_perm),
  1177	        yVertex=reorder_1d(mesh.yVertex, vert_perm),
  1178	        zVertex=reorder_1d(mesh.zVertex, vert_perm),
  1179	        # Connectivity: reorder columns AND remap values
  1180	        cellsOnEdge=remap_conn(reorder_col(mesh.cellsOnEdge, edge_perm), cell_inv),
  1181	        edgesOnCell=remap_conn(reorder_col(mesh.edgesOnCell, cell_perm), edge_inv),
  1182	        verticesOnCell=remap_conn(reorder_col(mesh.verticesOnCell, cell_perm), vert_inv),
  1183	        verticesOnEdge=remap_conn(reorder_col(mesh.verticesOnEdge, edge_perm), vert_inv),
  1184	        edgesOnVertex=remap_conn(reorder_col(mesh.edgesOnVertex, vert_perm), edge_inv),
  1185	        cellsOnVertex=remap_conn(reorder_col(mesh.cellsOnVertex, vert_perm), cell_inv),
  1186	        cellsOnCell=remap_conn(reorder_col(mesh.cellsOnCell, cell_perm), cell_inv),
  1187	        edgesOnEdge=remap_conn(reorder_col(mesh.edgesOnEdge, edge_perm), edge_inv),
  1188	        nEdgesOnCell=reorder_1d(mesh.nEdgesOnCell, cell_perm),
  1189	        nEdgesOnEdge=reorder_1d(mesh.nEdgesOnEdge, edge_perm),
  1190	        # Geometry (reorder by entity)
  1191	        areaCell=reorder_1d(mesh.areaCell, cell_perm),
  1192	        areaTriangle=reorder_1d(mesh.areaTriangle, vert_perm),
  1193	        dcEdge=reorder_1d(mesh.dcEdge, edge_perm),
  1194	        dvEdge=reorder_1d(mesh.dvEdge, edge_perm),
  1195	        angleEdge=reorder_1d(mesh.angleEdge, edge_perm),
  1196	        # Weights and signs (reorder columns by entity)
  1197	        weightsOnEdge=reorder_col(mesh.weightsOnEdge, edge_perm),
  1198	        kiteAreasOnVertex=reorder_col(mesh.kiteAreasOnVertex, vert_perm),
  1199	        fEdge=reorder_1d(mesh.fEdge, edge_perm),
  1200	        fVertex=reorder_1d(mesh.fVertex, vert_perm),
  1201	        edgeSignOnCell=reorder_col(mesh.edgeSignOnCell, cell_perm),
  1202	        edgeSignOnVertex=reorder_col(mesh.edgeSignOnVertex, vert_perm),
  1203	        meshDensity=reorder_1d(mesh.meshDensity, cell_perm),
  1204	        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
  1205	                             else reorder_1d(mesh.subgrid_topo_stddev,
  1206	                                             cell_perm)),
  1207	        land_frac=(None if mesh.land_frac is None
  1208	                   else reorder_1d(mesh.land_frac, cell_perm)),
  1209	    )
  1210	
  1211	    # --- Pad so that nCells and nEdges are divisible by n_devices ---
  1212	    return _pad_voronoi_for_sharding(reordered, n_devices)
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

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1760,1995p'; nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '650,1020p'; rg -n -C 3 \"_pad_voronoi_for_sharding|def _pad|pad_cells|pad_edges\" packages/core/legoesm/parallel/voronoi_partition.py; nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '800,1010p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 172ms:
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
  1991	# serialized loop-fusion kernels that the HLO frame table resolves to the
  1992	# RK pytree_axpy (pytree_ops.py), ~10.9 ms/step in total (nsys 26479922,
  1993	# HLO 26480096), and the barrier removes most of that: same-day ladder
  1994	# np4 17.78 -> 14.10 ms (-20.7%) with np2 +1.2% (noise) and np8 0.0%
  1995	# (jobs 26486123 dead-gate vs 26486163). An UNgated barrier regressed np2
   650	        sorted(int(v) for v in all_local_verts if vertex_owner_all[v] != rank),
   651	        dtype=np.int64,
   652	    )
   653	    local_vertices = np.concatenate([owned_vertices, halo_vertices])
   654	
   655	    vertex_g2l = np.full(mesh.nVertices, -1, dtype=np.int64)
   656	    for i, g in enumerate(local_vertices):
   657	        vertex_g2l[g] = i
   658	
   659	    # ------------------------------------------------------------------
   660	    # Communication schedules
   661	    # ------------------------------------------------------------------
   662	    # Recv side: group halo entities by their owner rank.
   663	    cell_recv = _group_by_owner(halo_cells, cell_owner)
   664	    edge_recv = _group_by_owner(halo_edges, edge_owner_all)
   665	    vertex_recv = _group_by_owner(halo_vertices, vertex_owner_all)
   666	
   667	    neighbor_ranks = sorted(cell_recv.keys())
   668	
   669	    # For each neighbor rank R, precompute which cells are in R's local
   670	    # domain (owned + halo).  This lets us determine which of our owned
   671	    # edges/vertices R needs as halo.
   672	    # Candidate ranks for the SEND schedule must be a SUPERSET of the cell-recv
   673	    # neighbours.  A rank can share only an EDGE or VERTEX boundary with me — it
   674	    # holds an edge/vertex I own in its halo — without its cell-halo reaching my
   675	    # cells, so it is absent from `neighbor_ranks` (= cell-recv owners) and the
   676	    # cell-neighbour-only `cell_to_nbr` would never mark it as needing that edge:
   677	    # I would not send, its blocking sendrecv to me would hang.  This is the
   678	    # np>=64 multi-node deadlock (asymmetric edge schedule; cells were fine).
   679	    # An edge/vertex spans exactly one cell-ring beyond the cell halo, so owners
   680	    # of cells within (halo_depth + 1) rings of my owned cells are a provably
   681	    # sufficient superset (an edge I own has one cell of mine and one neighbour
   682	    # cell; any rank needing it is within halo_depth of that neighbour cell,
   683	    # i.e. within halo_depth + 1 of my cell).
   684	    send_candidate_cells = compute_halo_cells(
   685	        cell_owner, cellsOnCell, mesh.maxEdges, rank, halo_depth + 1,
   686	    )
   687	    cell_to_nbr_ranks = sorted(
   688	        (set(neighbor_ranks)
   689	         | {int(cell_owner[c]) for c in send_candidate_cells})
   690	        - {rank}
   691	    )
   692	    cell_to_nbr: dict[int, set[int]] = {}
   693	    for R in cell_to_nbr_ranks:
   694	        R_owned = set(np.where(cell_owner == R)[0].tolist())
   695	        R_halo = compute_halo_cells(
   696	            cell_owner, cellsOnCell, mesh.maxEdges, R, halo_depth,
   697	        )
   698	        for c in R_owned | R_halo:
   699	            cell_to_nbr.setdefault(c, set()).add(R)
   700	
   701	    # --- Cell send ---
   702	    cell_send: dict[int, list[int]] = {r: [] for r in neighbor_ranks}
   703	    for c in owned_cells:
   704	        c_int = int(c)
   705	        for r in cell_to_nbr.get(c_int, ()):
   706	            if r != rank:
   707	                cell_send.setdefault(r, []).append(c_int)
   708	    cell_send = {r: sorted(set(v)) for r, v in cell_send.items()}
   709	
   710	    # --- Edge send ---
   711	    edge_send: dict[int, list[int]] = {r: [] for r in neighbor_ranks}
   712	    for e in owned_edges:
   713	        c1, c2 = int(cellsOnEdge[0, e]), int(cellsOnEdge[1, e])
   714	        nbrs: set[int] = set()
   715	        nbrs |= cell_to_nbr.get(c1, set())
   716	        nbrs |= cell_to_nbr.get(c2, set())
   717	        for r in nbrs:
   718	            if r != rank:
   719	                edge_send.setdefault(r, []).append(int(e))
   720	    edge_send = {r: sorted(set(v)) for r, v in edge_send.items()}
   721	
   722	    # --- Vertex send ---
   723	    vert_send: dict[int, list[int]] = {r: [] for r in neighbor_ranks}
   724	    for v in owned_vertices:
   725	        nbrs_v: set[int] = set()
   726	        for k in range(mesh.vertexDegree):
   727	            c = int(cov_np[k, v])
   728	            if c >= 0:
   729	                nbrs_v |= cell_to_nbr.get(c, set())
   730	        for r in nbrs_v:
   731	            if r != rank:
   732	                vert_send.setdefault(r, []).append(int(v))
   733	    vert_send = {r: sorted(set(v)) for r, v in vert_send.items()}
   734	
   735	    # Assemble schedules.
   736	    cell_comm = _assemble_schedule(
   737	        cell_recv, cell_send, cell_g2l, neighbor_ranks)
   738	
   739	    edge_nbrs = sorted(
   740	        set(edge_recv.keys())
   741	        | {k for k, v in edge_send.items() if v}
   742	    )
   743	    edge_comm = _assemble_schedule(
   744	        edge_recv, edge_send, edge_g2l, edge_nbrs)
   745	
   746	    vert_nbrs = sorted(
   747	        set(vertex_recv.keys())
   748	        | {k for k, v in vert_send.items() if v}
   749	    )
   750	    vertex_comm = _assemble_schedule(
   751	        vertex_recv, vert_send, vertex_g2l, vert_nbrs)
   752	
   753	    return VoronoiPartition(
   754	        rank=rank,
   755	        n_ranks=n_ranks,
   756	        nCells_global=mesh.nCells,
   757	        nEdges_global=mesh.nEdges,
   758	        nVertices_global=mesh.nVertices,
   759	        n_owned_cells=len(owned_cells),
   760	        n_owned_edges=len(owned_edges),
   761	        n_owned_vertices=len(owned_vertices),
   762	        n_local_cells=len(local_cells),
   763	        n_local_edges=len(local_edges),
   764	        n_local_vertices=len(local_vertices),
   765	        local_cells=local_cells,
   766	        local_edges=local_edges,
   767	        local_vertices=local_vertices,
   768	        cell_g2l=cell_g2l,
   769	        edge_g2l=edge_g2l,
   770	        vertex_g2l=vertex_g2l,
   771	        cell_comm=cell_comm,
   772	        edge_comm=edge_comm,
   773	        vertex_comm=vertex_comm,
   774	    )
   775	
   776	
   777	# ============================================================================
   778	# Local mesh construction
   779	# ============================================================================
   780	
   781	def build_local_mesh(
   782	    mesh: VoronoiMesh,
   783	    partition: VoronoiPartition,
   784	) -> VoronoiMesh:
   785	    """Build a VoronoiMesh with local indices from a partition.
   786	
   787	    The returned mesh uses local (0-based) indices in its connectivity
   788	    arrays.  Entries that reference entities outside the local domain
   789	    are set to -1 (TRiSK operators already mask these).
   790	
   791	    Parameters
   792	    ----------
   793	    mesh : VoronoiMesh
   794	        Global mesh.
   795	    partition : VoronoiPartition
   796	
   797	    Returns
   798	    -------
   799	    VoronoiMesh
   800	        Local mesh with ``nCells = n_local_cells``, etc.
   801	    """
   802	    lc = partition.local_cells
   803	    le = partition.local_edges
   804	    lv = partition.local_vertices
   805	
   806	    # Global-to-local maps as JAX arrays for indexing.
   807	    c_g2l = jnp.array(partition.cell_g2l, dtype=jnp.int32)
   808	    e_g2l = jnp.array(partition.edge_g2l, dtype=jnp.int32)
   809	    v_g2l = jnp.array(partition.vertex_g2l, dtype=jnp.int32)
   810	
   811	    def remap(conn, local_ents, g2l_map):
   812	        """Select columns for local entities and remap values."""
   813	        sel = conn[:, local_ents]        # (K, n_local)
   814	        valid = sel >= 0
   815	        safe = jnp.maximum(sel, 0)
   816	        return jnp.where(valid, g2l_map[safe], -1).astype(jnp.int32)
   817	
   818	    return VoronoiMesh(
   819	        nCells=partition.n_local_cells,
   820	        nEdges=partition.n_local_edges,
   821	        nVertices=partition.n_local_vertices,
   822	        maxEdges=mesh.maxEdges,
   823	        vertexDegree=mesh.vertexDegree,
   824	        radius=mesh.radius,
   825	        # Cell coordinates
   826	        latCell=mesh.latCell[lc],
   827	        lonCell=mesh.lonCell[lc],
   828	        xCell=mesh.xCell[lc],
   829	        yCell=mesh.yCell[lc],
   830	        zCell=mesh.zCell[lc],
   831	        # Edge coordinates
   832	        latEdge=mesh.latEdge[le],
   833	        lonEdge=mesh.lonEdge[le],
   834	        xEdge=mesh.xEdge[le],
   835	        yEdge=mesh.yEdge[le],
   836	        zEdge=mesh.zEdge[le],
   837	        # Vertex coordinates
   838	        latVertex=mesh.latVertex[lv],
   839	        lonVertex=mesh.lonVertex[lv],
   840	        xVertex=mesh.xVertex[lv],
   841	        yVertex=mesh.yVertex[lv],
   842	        zVertex=mesh.zVertex[lv],
   843	        # Connectivity (column-select + value-remap)
   844	        cellsOnEdge=remap(mesh.cellsOnEdge, le, c_g2l),
   845	        edgesOnCell=remap(mesh.edgesOnCell, lc, e_g2l),
   846	        verticesOnCell=remap(mesh.verticesOnCell, lc, v_g2l),
   847	        verticesOnEdge=remap(mesh.verticesOnEdge, le, v_g2l),
   848	        edgesOnVertex=remap(mesh.edgesOnVertex, lv, e_g2l),
   849	        cellsOnVertex=remap(mesh.cellsOnVertex, lv, c_g2l),
   850	        cellsOnCell=remap(mesh.cellsOnCell, lc, c_g2l),
   851	        edgesOnEdge=remap(mesh.edgesOnEdge, le, e_g2l),
   852	        nEdgesOnCell=mesh.nEdgesOnCell[lc],
   853	        nEdgesOnEdge=mesh.nEdgesOnEdge[le],
   854	        # Geometry (column-select only)
   855	        areaCell=mesh.areaCell[lc],
   856	        areaTriangle=mesh.areaTriangle[lv],
   857	        dcEdge=mesh.dcEdge[le],
   858	        dvEdge=mesh.dvEdge[le],
   859	        angleEdge=mesh.angleEdge[le],
   860	        # Weights (column-select only, values are floats not indices)
   861	        weightsOnEdge=mesh.weightsOnEdge[:, le],
   862	        kiteAreasOnVertex=mesh.kiteAreasOnVertex[:, lv],
   863	        fEdge=mesh.fEdge[le],
   864	        fVertex=mesh.fVertex[lv],
   865	        edgeSignOnCell=mesh.edgeSignOnCell[:, lc],
   866	        edgeSignOnVertex=mesh.edgeSignOnVertex[:, lv],
   867	        meshDensity=mesh.meshDensity[lc],
   868	        # Optional per-cell surface fields: slice like any cell field so
   869	        # the LOCAL mesh keeps the SSO/land-fraction the global mesh
   870	        # carries (None stays None — legacy meshes unchanged).
   871	        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
   872	                             else mesh.subgrid_topo_stddev[lc]),
   873	        land_frac=(None if mesh.land_frac is None
   874	                   else mesh.land_frac[lc]),
   875	    )
   876	
   877	
   878	# ============================================================================
   879	# Convenience utilities
   880	# ============================================================================
   881	
   882	def scatter_to_local(
   883	    global_field: jnp.ndarray,
   884	    partition: VoronoiPartition,
   885	    entity: str = "cell",
   886	) -> jnp.ndarray:
   887	    """Extract local portion (owned + halo) of a global field.
   888	
   889	    Parameters
   890	    ----------
   891	    global_field : jax.Array
   892	        Full-mesh field, shape ``(nEntities, ...)`` or ``(nEntities,)``.
   893	    partition : VoronoiPartition
   894	    entity : str
   895	        ``"cell"``, ``"edge"``, or ``"vertex"``.
   896	
   897	    Returns
   898	    -------
   899	    jax.Array, shape ``(n_local_entities, ...)``
   900	    """
   901	    idx = {
   902	        "cell": partition.local_cells,
   903	        "edge": partition.local_edges,
   904	        "vertex": partition.local_vertices,
   905	    }[entity]
   906	    return global_field[idx]
   907	
   908	
   909	# ============================================================================
   910	# Mesh padding for even sharding
   911	# ============================================================================
   912	
   913	def _pad_voronoi_for_sharding(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
   914	    """Pad cell/edge arrays so their sizes are divisible by *n_devices*.
   915	
   916	    Adds ghost cells/edges that are inert in physics:
   917	    - Ghost cells: ``areaCell=1`` (avoids 0/0 NaN in divergence), all
   918	      connectivity = -1 (masked by operators), signs/weights = 0.
   919	    - Ghost edges: ``dvEdge=0`` (zero flux contribution), ``dcEdge=1``
   920	      (avoids 0/0 in gradient), ``cellsOnEdge=[0,0]`` (valid references
   921	      for unmasked operators like ``gradient_edge`` and ``cell_to_edge_avg``).
   922	
   923	    Returns *mesh* unchanged when no padding is required.
   924	    """
   925	    pad_cells = (-mesh.nCells) % n_devices
   926	    pad_edges = (-mesh.nEdges) % n_devices
   927	
   928	    if pad_cells == 0 and pad_edges == 0:
   929	        return mesh
   930	
   931	    # --- helpers ---
   932	    def pad_1d(arr, n_pad, fill=0.0):
   933	        if n_pad == 0:
   934	            return arr
   935	        return jnp.concatenate([arr, jnp.full((n_pad,), fill, dtype=arr.dtype)])
   936	
   937	    def pad_2d_col(arr, n_pad, fill=0):
   938	        """Pad along axis 1 (entity axis for (K, nEntities) layout)."""
   939	        if n_pad == 0:
   940	            return arr
   941	        K = arr.shape[0]
   942	        return jnp.concatenate(
   943	            [arr, jnp.full((K, n_pad), fill, dtype=arr.dtype)], axis=1,
   944	        )
   945	
   946	    return VoronoiMesh(
   947	        # --- dimensions ---
   948	        nCells=mesh.nCells + pad_cells,
   949	        nEdges=mesh.nEdges + pad_edges,
   950	        nVertices=mesh.nVertices,
   951	        maxEdges=mesh.maxEdges,
   952	        vertexDegree=mesh.vertexDegree,
   953	        radius=mesh.radius,
   954	        # --- cell coordinates (ghost at origin) ---
   955	        latCell=pad_1d(mesh.latCell, pad_cells, 0.0),
   956	        lonCell=pad_1d(mesh.lonCell, pad_cells, 0.0),
   957	        xCell=pad_1d(mesh.xCell, pad_cells, 0.0),
   958	        yCell=pad_1d(mesh.yCell, pad_cells, 0.0),
   959	        zCell=pad_1d(mesh.zCell, pad_cells, 0.0),
   960	        # --- edge coordinates (ghost at origin) ---
   961	        latEdge=pad_1d(mesh.latEdge, pad_edges, 0.0),
   962	        lonEdge=pad_1d(mesh.lonEdge, pad_edges, 0.0),
   963	        xEdge=pad_1d(mesh.xEdge, pad_edges, 0.0),
   964	        yEdge=pad_1d(mesh.yEdge, pad_edges, 0.0),
   965	        zEdge=pad_1d(mesh.zEdge, pad_edges, 0.0),
   966	        # --- vertex coordinates (unchanged) ---
   967	        latVertex=mesh.latVertex,
   968	        lonVertex=mesh.lonVertex,
   969	        xVertex=mesh.xVertex,
   970	        yVertex=mesh.yVertex,
   971	        zVertex=mesh.zVertex,
   972	        # --- connectivity ---
   973	        # cellsOnEdge: operators (gradient_edge, cell_to_edge_avg) index
   974	        # directly without masking, so ghost edges need valid cell refs.
   975	        cellsOnEdge=pad_2d_col(mesh.cellsOnEdge, pad_edges, fill=0),
   976	        edgesOnCell=pad_2d_col(mesh.edgesOnCell, pad_cells, fill=-1),
   977	        verticesOnCell=pad_2d_col(mesh.verticesOnCell, pad_cells, fill=-1),
   978	        verticesOnEdge=pad_2d_col(mesh.verticesOnEdge, pad_edges, fill=0),
   979	        edgesOnVertex=mesh.edgesOnVertex,  # vertex-indexed, unchanged
   980	        cellsOnVertex=mesh.cellsOnVertex,  # vertex-indexed, unchanged
   981	        cellsOnCell=pad_2d_col(mesh.cellsOnCell, pad_cells, fill=-1),
   982	        edgesOnEdge=pad_2d_col(mesh.edgesOnEdge, pad_edges, fill=-1),
   983	        nEdgesOnCell=pad_1d(mesh.nEdgesOnCell, pad_cells, fill=0),
   984	        nEdgesOnEdge=pad_1d(mesh.nEdgesOnEdge, pad_edges, fill=0),
   985	        # --- geometry ---
   986	        # areaCell=1 for ghosts avoids 0/0 NaN in divergence (numerator is
   987	        # exactly zero because all connectivity = -1 and signs = 0).
   988	        areaCell=pad_1d(mesh.areaCell, pad_cells, fill=1.0),
   989	        areaTriangle=mesh.areaTriangle,  # vertex-indexed, unchanged
   990	        dcEdge=pad_1d(mesh.dcEdge, pad_edges, fill=1.0),  # avoid /0
   991	        dvEdge=pad_1d(mesh.dvEdge, pad_edges, fill=0.0),  # zero flux
   992	        angleEdge=pad_1d(mesh.angleEdge, pad_edges, fill=0.0),
   993	        # --- weights and signs ---
   994	        weightsOnEdge=pad_2d_col(mesh.weightsOnEdge, pad_edges, fill=0.0),
   995	        kiteAreasOnVertex=mesh.kiteAreasOnVertex,  # vertex-indexed
   996	        fEdge=pad_1d(mesh.fEdge, pad_edges, fill=0.0),
   997	        fVertex=mesh.fVertex,  # vertex-indexed, unchanged
   998	        edgeSignOnCell=pad_2d_col(mesh.edgeSignOnCell, pad_cells, fill=0.0),
   999	        edgeSignOnVertex=mesh.edgeSignOnVertex,  # vertex-indexed
  1000	        meshDensity=pad_1d(mesh.meshDensity, pad_cells, fill=0.0),
  1001	        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
  1002	                             else pad_1d(mesh.subgrid_topo_stddev,
  1003	                                         pad_cells, fill=0.0)),
  1004	        land_frac=(None if mesh.land_frac is None
  1005	                   else pad_1d(mesh.land_frac, pad_cells, fill=0.0)),
  1006	    )
  1007	
  1008	
  1009	# ============================================================================
  1010	# Mesh reordering for JAX SPMD sharding
  1011	# ============================================================================
  1012	
  1013	def resolve_sharding_partition_method(method: str) -> str:
  1014	    """Concrete ownership for the SPMD/ppermute path (``auto`` -> ``sfc``).
  1015	
  1016	    Separate from :func:`resolve_partition_method`, whose ``auto`` prefers
  1017	    METIS: METIS minimizes edge CUT, while this path is bound by the number of
  1018	    sequential halo exchanges, and measured they move oppositely (subdiv-8 at
  1019	    128 devices: sfc 14 rounds, metis 19).  ONE definition, so a scorer that
  1020	    reports which ownership ran cannot drift from what the reorder does.
910-# Mesh padding for even sharding
911-# ============================================================================
912-
913:def _pad_voronoi_for_sharding(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
914-    """Pad cell/edge arrays so their sizes are divisible by *n_devices*.
915-
916-    Adds ghost cells/edges that are inert in physics:
--
922-
923-    Returns *mesh* unchanged when no padding is required.
924-    """
925:    pad_cells = (-mesh.nCells) % n_devices
926:    pad_edges = (-mesh.nEdges) % n_devices
927-
928:    if pad_cells == 0 and pad_edges == 0:
929-        return mesh
930-
931-    # --- helpers ---
--
945-
946-    return VoronoiMesh(
947-        # --- dimensions ---
948:        nCells=mesh.nCells + pad_cells,
949:        nEdges=mesh.nEdges + pad_edges,
950-        nVertices=mesh.nVertices,
951-        maxEdges=mesh.maxEdges,
952-        vertexDegree=mesh.vertexDegree,
953-        radius=mesh.radius,
954-        # --- cell coordinates (ghost at origin) ---
955:        latCell=pad_1d(mesh.latCell, pad_cells, 0.0),
956:        lonCell=pad_1d(mesh.lonCell, pad_cells, 0.0),
957:        xCell=pad_1d(mesh.xCell, pad_cells, 0.0),
958:        yCell=pad_1d(mesh.yCell, pad_cells, 0.0),
959:        zCell=pad_1d(mesh.zCell, pad_cells, 0.0),
960-        # --- edge coordinates (ghost at origin) ---
961:        latEdge=pad_1d(mesh.latEdge, pad_edges, 0.0),
962:        lonEdge=pad_1d(mesh.lonEdge, pad_edges, 0.0),
963:        xEdge=pad_1d(mesh.xEdge, pad_edges, 0.0),
964:        yEdge=pad_1d(mesh.yEdge, pad_edges, 0.0),
965:        zEdge=pad_1d(mesh.zEdge, pad_edges, 0.0),
966-        # --- vertex coordinates (unchanged) ---
967-        latVertex=mesh.latVertex,
968-        lonVertex=mesh.lonVertex,
--
972-        # --- connectivity ---
973-        # cellsOnEdge: operators (gradient_edge, cell_to_edge_avg) index
974-        # directly without masking, so ghost edges need valid cell refs.
975:        cellsOnEdge=pad_2d_col(mesh.cellsOnEdge, pad_edges, fill=0),
976:        edgesOnCell=pad_2d_col(mesh.edgesOnCell, pad_cells, fill=-1),
977:        verticesOnCell=pad_2d_col(mesh.verticesOnCell, pad_cells, fill=-1),
978:        verticesOnEdge=pad_2d_col(mesh.verticesOnEdge, pad_edges, fill=0),
979-        edgesOnVertex=mesh.edgesOnVertex,  # vertex-indexed, unchanged
980-        cellsOnVertex=mesh.cellsOnVertex,  # vertex-indexed, unchanged
981:        cellsOnCell=pad_2d_col(mesh.cellsOnCell, pad_cells, fill=-1),
982:        edgesOnEdge=pad_2d_col(mesh.edgesOnEdge, pad_edges, fill=-1),
983:        nEdgesOnCell=pad_1d(mesh.nEdgesOnCell, pad_cells, fill=0),
984:        nEdgesOnEdge=pad_1d(mesh.nEdgesOnEdge, pad_edges, fill=0),
985-        # --- geometry ---
986-        # areaCell=1 for ghosts avoids 0/0 NaN in divergence (numerator is
987-        # exactly zero because all connectivity = -1 and signs = 0).
988:        areaCell=pad_1d(mesh.areaCell, pad_cells, fill=1.0),
989-        areaTriangle=mesh.areaTriangle,  # vertex-indexed, unchanged
990:        dcEdge=pad_1d(mesh.dcEdge, pad_edges, fill=1.0),  # avoid /0
991:        dvEdge=pad_1d(mesh.dvEdge, pad_edges, fill=0.0),  # zero flux
992:        angleEdge=pad_1d(mesh.angleEdge, pad_edges, fill=0.0),
993-        # --- weights and signs ---
994:        weightsOnEdge=pad_2d_col(mesh.weightsOnEdge, pad_edges, fill=0.0),
995-        kiteAreasOnVertex=mesh.kiteAreasOnVertex,  # vertex-indexed
996:        fEdge=pad_1d(mesh.fEdge, pad_edges, fill=0.0),
997-        fVertex=mesh.fVertex,  # vertex-indexed, unchanged
998:        edgeSignOnCell=pad_2d_col(mesh.edgeSignOnCell, pad_cells, fill=0.0),
999-        edgeSignOnVertex=mesh.edgeSignOnVertex,  # vertex-indexed
1000:        meshDensity=pad_1d(mesh.meshDensity, pad_cells, fill=0.0),
1001-        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
1002-                             else pad_1d(mesh.subgrid_topo_stddev,
1003:                                         pad_cells, fill=0.0)),
1004-        land_frac=(None if mesh.land_frac is None
1005:                   else pad_1d(mesh.land_frac, pad_cells, fill=0.0)),
1006-    )
1007-
1008-
--
1209-    )
1210-
1211-    # --- Pad so that nCells and nEdges are divisible by n_devices ---
1212:    return _pad_voronoi_for_sharding(reordered, n_devices)
   800	        Local mesh with ``nCells = n_local_cells``, etc.
   801	    """
   802	    lc = partition.local_cells
   803	    le = partition.local_edges
   804	    lv = partition.local_vertices
   805	
   806	    # Global-to-local maps as JAX arrays for indexing.
   807	    c_g2l = jnp.array(partition.cell_g2l, dtype=jnp.int32)
   808	    e_g2l = jnp.array(partition.edge_g2l, dtype=jnp.int32)
   809	    v_g2l = jnp.array(partition.vertex_g2l, dtype=jnp.int32)
   810	
   811	    def remap(conn, local_ents, g2l_map):
   812	        """Select columns for local entities and remap values."""
   813	        sel = conn[:, local_ents]        # (K, n_local)
   814	        valid = sel >= 0
   815	        safe = jnp.maximum(sel, 0)
   816	        return jnp.where(valid, g2l_map[safe], -1).astype(jnp.int32)
   817	
   818	    return VoronoiMesh(
   819	        nCells=partition.n_local_cells,
   820	        nEdges=partition.n_local_edges,
   821	        nVertices=partition.n_local_vertices,
   822	        maxEdges=mesh.maxEdges,
   823	        vertexDegree=mesh.vertexDegree,
   824	        radius=mesh.radius,
   825	        # Cell coordinates
   826	        latCell=mesh.latCell[lc],
   827	        lonCell=mesh.lonCell[lc],
   828	        xCell=mesh.xCell[lc],
   829	        yCell=mesh.yCell[lc],
   830	        zCell=mesh.zCell[lc],
   831	        # Edge coordinates
   832	        latEdge=mesh.latEdge[le],
   833	        lonEdge=mesh.lonEdge[le],
   834	        xEdge=mesh.xEdge[le],
   835	        yEdge=mesh.yEdge[le],
   836	        zEdge=mesh.zEdge[le],
   837	        # Vertex coordinates
   838	        latVertex=mesh.latVertex[lv],
   839	        lonVertex=mesh.lonVertex[lv],
   840	        xVertex=mesh.xVertex[lv],
   841	        yVertex=mesh.yVertex[lv],
   842	        zVertex=mesh.zVertex[lv],
   843	        # Connectivity (column-select + value-remap)
   844	        cellsOnEdge=remap(mesh.cellsOnEdge, le, c_g2l),
   845	        edgesOnCell=remap(mesh.edgesOnCell, lc, e_g2l),
   846	        verticesOnCell=remap(mesh.verticesOnCell, lc, v_g2l),
   847	        verticesOnEdge=remap(mesh.verticesOnEdge, le, v_g2l),
   848	        edgesOnVertex=remap(mesh.edgesOnVertex, lv, e_g2l),
   849	        cellsOnVertex=remap(mesh.cellsOnVertex, lv, c_g2l),
   850	        cellsOnCell=remap(mesh.cellsOnCell, lc, c_g2l),
   851	        edgesOnEdge=remap(mesh.edgesOnEdge, le, e_g2l),
   852	        nEdgesOnCell=mesh.nEdgesOnCell[lc],
   853	        nEdgesOnEdge=mesh.nEdgesOnEdge[le],
   854	        # Geometry (column-select only)
   855	        areaCell=mesh.areaCell[lc],
   856	        areaTriangle=mesh.areaTriangle[lv],
   857	        dcEdge=mesh.dcEdge[le],
   858	        dvEdge=mesh.dvEdge[le],
   859	        angleEdge=mesh.angleEdge[le],
   860	        # Weights (column-select only, values are floats not indices)
   861	        weightsOnEdge=mesh.weightsOnEdge[:, le],
   862	        kiteAreasOnVertex=mesh.kiteAreasOnVertex[:, lv],
   863	        fEdge=mesh.fEdge[le],
   864	        fVertex=mesh.fVertex[lv],
   865	        edgeSignOnCell=mesh.edgeSignOnCell[:, lc],
   866	        edgeSignOnVertex=mesh.edgeSignOnVertex[:, lv],
   867	        meshDensity=mesh.meshDensity[lc],
   868	        # Optional per-cell surface fields: slice like any cell field so
   869	        # the LOCAL mesh keeps the SSO/land-fraction the global mesh
   870	        # carries (None stays None — legacy meshes unchanged).
   871	        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
   872	                             else mesh.subgrid_topo_stddev[lc]),
   873	        land_frac=(None if mesh.land_frac is None
   874	                   else mesh.land_frac[lc]),
   875	    )
   876	
   877	
   878	# ============================================================================
   879	# Convenience utilities
   880	# ============================================================================
   881	
   882	def scatter_to_local(
   883	    global_field: jnp.ndarray,
   884	    partition: VoronoiPartition,
   885	    entity: str = "cell",
   886	) -> jnp.ndarray:
   887	    """Extract local portion (owned + halo) of a global field.
   888	
   889	    Parameters
   890	    ----------
   891	    global_field : jax.Array
   892	        Full-mesh field, shape ``(nEntities, ...)`` or ``(nEntities,)``.
   893	    partition : VoronoiPartition
   894	    entity : str
   895	        ``"cell"``, ``"edge"``, or ``"vertex"``.
   896	
   897	    Returns
   898	    -------
   899	    jax.Array, shape ``(n_local_entities, ...)``
   900	    """
   901	    idx = {
   902	        "cell": partition.local_cells,
   903	        "edge": partition.local_edges,
   904	        "vertex": partition.local_vertices,
   905	    }[entity]
   906	    return global_field[idx]
   907	
   908	
   909	# ============================================================================
   910	# Mesh padding for even sharding
   911	# ============================================================================
   912	
   913	def _pad_voronoi_for_sharding(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
   914	    """Pad cell/edge arrays so their sizes are divisible by *n_devices*.
   915	
   916	    Adds ghost cells/edges that are inert in physics:
   917	    - Ghost cells: ``areaCell=1`` (avoids 0/0 NaN in divergence), all
   918	      connectivity = -1 (masked by operators), signs/weights = 0.
   919	    - Ghost edges: ``dvEdge=0`` (zero flux contribution), ``dcEdge=1``
   920	      (avoids 0/0 in gradient), ``cellsOnEdge=[0,0]`` (valid references
   921	      for unmasked operators like ``gradient_edge`` and ``cell_to_edge_avg``).
   922	
   923	    Returns *mesh* unchanged when no padding is required.
   924	    """
   925	    pad_cells = (-mesh.nCells) % n_devices
   926	    pad_edges = (-mesh.nEdges) % n_devices
   927	
   928	    if pad_cells == 0 and pad_edges == 0:
   929	        return mesh
   930	
   931	    # --- helpers ---
   932	    def pad_1d(arr, n_pad, fill=0.0):
   933	        if n_pad == 0:
   934	            return arr
   935	        return jnp.concatenate([arr, jnp.full((n_pad,), fill, dtype=arr.dtype)])
   936	
   937	    def pad_2d_col(arr, n_pad, fill=0):
   938	        """Pad along axis 1 (entity axis for (K, nEntities) layout)."""
   939	        if n_pad == 0:
   940	            return arr
   941	        K = arr.shape[0]
   942	        return jnp.concatenate(
   943	            [arr, jnp.full((K, n_pad), fill, dtype=arr.dtype)], axis=1,
   944	        )
   945	
   946	    return VoronoiMesh(
   947	        # --- dimensions ---
   948	        nCells=mesh.nCells + pad_cells,
   949	        nEdges=mesh.nEdges + pad_edges,
   950	        nVertices=mesh.nVertices,
   951	        maxEdges=mesh.maxEdges,
   952	        vertexDegree=mesh.vertexDegree,
   953	        radius=mesh.radius,
   954	        # --- cell coordinates (ghost at origin) ---
   955	        latCell=pad_1d(mesh.latCell, pad_cells, 0.0),
   956	        lonCell=pad_1d(mesh.lonCell, pad_cells, 0.0),
   957	        xCell=pad_1d(mesh.xCell, pad_cells, 0.0),
   958	        yCell=pad_1d(mesh.yCell, pad_cells, 0.0),
   959	        zCell=pad_1d(mesh.zCell, pad_cells, 0.0),
   960	        # --- edge coordinates (ghost at origin) ---
   961	        latEdge=pad_1d(mesh.latEdge, pad_edges, 0.0),
   962	        lonEdge=pad_1d(mesh.lonEdge, pad_edges, 0.0),
   963	        xEdge=pad_1d(mesh.xEdge, pad_edges, 0.0),
   964	        yEdge=pad_1d(mesh.yEdge, pad_edges, 0.0),
   965	        zEdge=pad_1d(mesh.zEdge, pad_edges, 0.0),
   966	        # --- vertex coordinates (unchanged) ---
   967	        latVertex=mesh.latVertex,
   968	        lonVertex=mesh.lonVertex,
   969	        xVertex=mesh.xVertex,
   970	        yVertex=mesh.yVertex,
   971	        zVertex=mesh.zVertex,
   972	        # --- connectivity ---
   973	        # cellsOnEdge: operators (gradient_edge, cell_to_edge_avg) index
   974	        # directly without masking, so ghost edges need valid cell refs.
   975	        cellsOnEdge=pad_2d_col(mesh.cellsOnEdge, pad_edges, fill=0),
   976	        edgesOnCell=pad_2d_col(mesh.edgesOnCell, pad_cells, fill=-1),
   977	        verticesOnCell=pad_2d_col(mesh.verticesOnCell, pad_cells, fill=-1),
   978	        verticesOnEdge=pad_2d_col(mesh.verticesOnEdge, pad_edges, fill=0),
   979	        edgesOnVertex=mesh.edgesOnVertex,  # vertex-indexed, unchanged
   980	        cellsOnVertex=mesh.cellsOnVertex,  # vertex-indexed, unchanged
   981	        cellsOnCell=pad_2d_col(mesh.cellsOnCell, pad_cells, fill=-1),
   982	        edgesOnEdge=pad_2d_col(mesh.edgesOnEdge, pad_edges, fill=-1),
   983	        nEdgesOnCell=pad_1d(mesh.nEdgesOnCell, pad_cells, fill=0),
   984	        nEdgesOnEdge=pad_1d(mesh.nEdgesOnEdge, pad_edges, fill=0),
   985	        # --- geometry ---
   986	        # areaCell=1 for ghosts avoids 0/0 NaN in divergence (numerator is
   987	        # exactly zero because all connectivity = -1 and signs = 0).
   988	        areaCell=pad_1d(mesh.areaCell, pad_cells, fill=1.0),
   989	        areaTriangle=mesh.areaTriangle,  # vertex-indexed, unchanged
   990	        dcEdge=pad_1d(mesh.dcEdge, pad_edges, fill=1.0),  # avoid /0
   991	        dvEdge=pad_1d(mesh.dvEdge, pad_edges, fill=0.0),  # zero flux
   992	        angleEdge=pad_1d(mesh.angleEdge, pad_edges, fill=0.0),
   993	        # --- weights and signs ---
   994	        weightsOnEdge=pad_2d_col(mesh.weightsOnEdge, pad_edges, fill=0.0),
   995	        kiteAreasOnVertex=mesh.kiteAreasOnVertex,  # vertex-indexed
   996	        fEdge=pad_1d(mesh.fEdge, pad_edges, fill=0.0),
   997	        fVertex=mesh.fVertex,  # vertex-indexed, unchanged
   998	        edgeSignOnCell=pad_2d_col(mesh.edgeSignOnCell, pad_cells, fill=0.0),
   999	        edgeSignOnVertex=mesh.edgeSignOnVertex,  # vertex-indexed
  1000	        meshDensity=pad_1d(mesh.meshDensity, pad_cells, fill=0.0),
  1001	        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
  1002	                             else pad_1d(mesh.subgrid_topo_stddev,
  1003	                                         pad_cells, fill=0.0)),
  1004	        land_frac=(None if mesh.land_frac is None
  1005	                   else pad_1d(mesh.land_frac, pad_cells, fill=0.0)),
  1006	    )
  1007	
  1008	
  1009	# ============================================================================
  1010	# Mesh reordering for JAX SPMD sharding

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1400,1665p'; rg -n -C 3 \"def make_voronoi_sharded_step|_build_voronoi_partition_infra|_build_ppermute_schedule|SPMD_HALO_DEPTH|halo_strategy|ppermute_cells_per_device_threshold|FORCE_ALLGATHER\" packages/core/legoesm/parallel/sharded_dynamics.py packages/ocean scripts tests | head -n 120; rg -n -C 3 \"create_voronoi_mesh\\(|lloyd_iterations\" packages/core/legoesm/grids/voronoi.py | head -n 160; rg -n \"spmd_schedule_cost\\(\" -S . -g '*.py' -g '*.sbatch' -g '*.md'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 2304ms:
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
  1421	        Each leaf has shape ``(n_dev, max_local_*)``.
  1422	    gather_cells : jnp.ndarray, (n_dev, max_local_cells)
  1423	    gather_edges : jnp.ndarray, (n_dev, max_local_edges)
  1424	    n_owned_cells : list[int]
  1425	    n_owned_edges : list[int]
  1426	    max_lc : int
  1427	    max_le : int
  1428	    partitions : list[VoronoiPartition]
  1429	        Per-device partition descriptors (for ppermute schedule building).
  1430	    cell_owner : np.ndarray, (nCells,)
  1431	        Cell ownership array.
  1432	    """
  1433	    import numpy as np
  1434	    from legoesm.parallel.voronoi_partition import (
  1435	        HaloCommSchedule,
  1436	        VoronoiPartition,
  1437	        build_local_mesh,
  1438	        compute_halo_cells,
  1439	    )
  1440	
  1441	    nCells = global_mesh.nCells
  1442	    nEdges = global_mesh.nEdges
  1443	    nVertices = global_mesh.nVertices
  1444	    cells_per = nCells // n_dev
  1445	    edges_per = nEdges // n_dev
  1446	    verts_per = nVertices // n_dev
  1447	
  1448	    # Convert mesh arrays to numpy for setup.
  1449	    gm_np = jax.tree.map(
  1450	        lambda x: np.asarray(x) if hasattr(x, "shape") else x,
  1451	        global_mesh,
  1452	    )
  1453	    cellsOnCell_np = np.asarray(gm_np.cellsOnCell)       # (maxEdges, nCells)
  1454	    cellsOnEdge_np = np.asarray(gm_np.cellsOnEdge)       # (2, nEdges)
  1455	    np.asarray(gm_np.cellsOnVertex)    # (vDeg, nVerts)
  1456	    verticesOnCell_np = np.asarray(gm_np.verticesOnCell)   # (maxEdges, nCells)
  1457	    verticesOnEdge_np = np.asarray(gm_np.verticesOnEdge)   # (2, nEdges)
  1458	    maxEdges = int(gm_np.maxEdges)
  1459	    int(gm_np.vertexDegree)
  1460	
  1461	    # Contiguous-block cell ownership (matches shard layout).
  1462	    cell_owner = np.repeat(np.arange(n_dev, dtype=np.int32), cells_per)
  1463	    if len(cell_owner) < nCells:
  1464	        cell_owner = np.concatenate([
  1465	            cell_owner,
  1466	            np.full(nCells - len(cell_owner), n_dev - 1, dtype=np.int32),
  1467	        ])
  1468	
  1469	    # Dummy comm schedule (not needed for shard_map path).
  1470	    _dummy_comm = HaloCommSchedule(
  1471	        neighbor_ranks=(), send_counts=(), recv_counts=(),
  1472	        send_idx=jnp.empty(0, dtype=jnp.int32),
  1473	        recv_idx=jnp.empty(0, dtype=jnp.int32),
  1474	    )
  1475	
  1476	    partitions = []
  1477	    local_meshes_raw = []
  1478	
  1479	    for rank in range(n_dev):
  1480	        # ----- Owned entities (contiguous blocks) ----- #
  1481	        c_start, c_end = rank * cells_per, (rank + 1) * cells_per
  1482	        e_start, e_end = rank * edges_per, (rank + 1) * edges_per
  1483	        v_start, v_end = rank * verts_per, (rank + 1) * verts_per
  1484	
  1485	        owned_cells = np.arange(c_start, c_end, dtype=np.int64)
  1486	        owned_edges = np.arange(e_start, e_end, dtype=np.int64)
  1487	        owned_vertices = np.arange(v_start, v_end, dtype=np.int64)
  1488	        owned_cells_set = set(owned_cells.tolist())
  1489	        set(owned_edges.tolist())
  1490	
  1491	        # ----- Halo cells: k-ring neighbours of owned cells ----- #
  1492	        halo_cells_set = compute_halo_cells(
  1493	            cell_owner, cellsOnCell_np, maxEdges, rank, halo_depth,
  1494	        )
  1495	        # Augment with the OTHER cell of every owned edge: on Voronoi
  1496	        # SCVT meshes the cellsOnCell adjacency *should* match the
  1497	        # cellsOnEdge connectivity, but the k-ring construction in
  1498	        # ``compute_halo_cells`` can miss a handful of cells at the
  1499	        # mesh boundary or near pentagons (owned edges whose far cell
  1500	        # is reachable via cellsOnEdge but whose hop chain through
  1501	        # cellsOnCell at depth <= halo_depth is broken by a -1 slot
  1502	        # or pentagon irregularity).  Without this augmentation, the
  1503	        # local mesh's cellsOnEdge has -1 entries for those edges
  1504	        # after remap → ``gradient_edge`` does ``phi[-1]`` (Python
  1505	        # last-element indexing!) and produces 1e76 garbage
  1506	        # within one SSP-RK3 step.  Iter-23 root-cause analysis.
  1507	        owned_edge_cells = cellsOnEdge_np[:, owned_edges].reshape(-1)
  1508	        owned_edge_cells = owned_edge_cells[owned_edge_cells >= 0]
  1509	        for c in owned_edge_cells:
  1510	            ic = int(c)
  1511	            if ic not in owned_cells_set:
  1512	                halo_cells_set.add(ic)
  1513	        # Iter-25/27/37: extend the halo so every halo cell's adjacent
  1514	        # edges have BOTH cells in local_cells.  Without this, halo
  1515	        # cells have some of their adjacent edges silently excluded by
  1516	        # the AND filter below; operator quantities at halo cells then
  1517	        # differ slightly from single-device, and owned-cell tendencies
  1518	        # that read those halo quantities inherit ~1% drift per step.
  1519	        # Two passes are sufficient for the dycore's longest operator
  1520	        # chain (depth 2: cell → edge → cell, twice for ∇⁴ in
  1521	        # hyperdiffusion) — see iter-37 docstring.
  1522	        owned_cells_arr = np.asarray(owned_cells, dtype=np.int64)
  1523	        _close_halo_under_cellsOnEdge(
  1524	            owned_cells_arr, halo_cells_set, cellsOnEdge_np, n_passes=2,
  1525	        )
  1526	        halo_cells = np.array(sorted(halo_cells_set), dtype=np.int64)
  1527	        local_cells = np.concatenate([owned_cells, halo_cells])
  1528	        set(local_cells.tolist())
  1529	
  1530	        # ----- Halo edges: edges where BOTH cellsOnEdge are in local_cells ----- #
  1531	        # Original Python loop over nEdges scaled poorly at MPAS resolutions
  1532	        # (1M+ cells); replace with a single ``np.isin`` on the cellsOnEdge
  1533	        # neighbour arrays so the whole partition setup is O(nEdges) numpy.
  1534	        #
  1535	        # CRITICAL: use AND, not OR.  Including an edge whose only one
  1536	        # neighbour is in local_cells leaves the other neighbour at -1
  1537	        # after the global→local remap (``cell_g2l[non_local] == -1``);
  1538	        # downstream operators like ``gradient_edge(phi, mesh)`` then
  1539	        # do ``phi[c1=-1]`` which is Python's last-element indexing
  1540	        # and produces garbage, blowing the SSP-RK3 step into 1e76
  1541	        # territory after a single iteration (iter-21 / iter-23 finding).
  1542	        # Using AND keeps the cellsOnEdge connectivity fully valid in the
  1543	        # local mesh; halo cells that lack some of their edges due to
  1544	        # the cut do not break correctness because the dycore only
  1545	        # uses tendencies on OWNED cells (halo tendencies are discarded
  1546	        # by the ``return`` in ``_local_tendency``).
  1547	        local_cells_arr = local_cells
  1548	        edge_in_local = np.isin(cellsOnEdge_np[0], local_cells_arr) & np.isin(
  1549	            cellsOnEdge_np[1], local_cells_arr,
  1550	        )
  1551	        local_edges_arr = np.flatnonzero(edge_in_local).astype(np.int64)
  1552	        # Sanity: every owned edge must satisfy the both-sides-local test
  1553	        # (an owned edge has at least one cell in owned_cells; the cell-
  1554	        # halo of depth >=1 covers the other side).  Surface a clear
  1555	        # error if a future mesh ordering change breaks that invariant.
  1556	        if not np.isin(owned_edges, local_edges_arr).all():
  1557	            missing = np.setdiff1d(owned_edges, local_edges_arr)
  1558	            raise RuntimeError(
  1559	                f"Voronoi partition rank={rank}: {len(missing)} owned "
  1560	                f"edges have a neighbour cell outside the halo-depth="
  1561	                f"{halo_depth} cell halo.  Increase halo_depth or check "
  1562	                f"the mesh ordering produced by reorder_voronoi_for_sharding."
  1563	            )
  1564	        halo_edges = np.setdiff1d(
  1565	            local_edges_arr, owned_edges, assume_unique=True,
  1566	        )
  1567	        local_edges = np.concatenate([owned_edges, halo_edges])
  1568	
  1569	        # ----- Halo vertices: vertices connected to local cells/edges ----- #
  1570	        # Vectorised gather: collect verticesOnCell over local cells and
  1571	        # verticesOnEdge over local edges in one pass each, then dedupe.
  1572	        verts_from_cells = verticesOnCell_np[:, local_cells].reshape(-1)
  1573	        verts_from_edges = verticesOnEdge_np[:, local_edges].reshape(-1)
  1574	        candidate_verts = np.concatenate([verts_from_cells, verts_from_edges])
  1575	        valid = (candidate_verts >= 0) & (candidate_verts < nVertices)
  1576	        local_verts_arr = np.unique(candidate_verts[valid]).astype(np.int64)
  1577	        halo_vertices = np.setdiff1d(
  1578	            local_verts_arr, owned_vertices, assume_unique=True,
  1579	        )
  1580	        local_vertices = np.concatenate([owned_vertices, halo_vertices])
  1581	
  1582	        # ----- Global-to-local maps ----- #
  1583	        cell_g2l = np.full(nCells, -1, dtype=np.int64)
  1584	        for i, g in enumerate(local_cells):
  1585	            cell_g2l[g] = i
  1586	        edge_g2l = np.full(nEdges, -1, dtype=np.int64)
  1587	        for i, g in enumerate(local_edges):
  1588	            edge_g2l[g] = i
  1589	        vertex_g2l = np.full(nVertices, -1, dtype=np.int64)
  1590	        for i, g in enumerate(local_vertices):
  1591	            vertex_g2l[g] = i
  1592	
  1593	        part = VoronoiPartition(
  1594	            rank=rank,
  1595	            n_ranks=n_dev,
  1596	            nCells_global=nCells,
  1597	            nEdges_global=nEdges,
  1598	            nVertices_global=nVertices,
  1599	            n_owned_cells=len(owned_cells),
  1600	            n_owned_edges=len(owned_edges),
  1601	            n_owned_vertices=len(owned_vertices),
  1602	            n_local_cells=len(local_cells),
  1603	            n_local_edges=len(local_edges),
  1604	            n_local_vertices=len(local_vertices),
  1605	            local_cells=local_cells,
  1606	            local_edges=local_edges,
  1607	            local_vertices=local_vertices,
  1608	            cell_g2l=cell_g2l,
  1609	            edge_g2l=edge_g2l,
  1610	            vertex_g2l=vertex_g2l,
  1611	            cell_comm=_dummy_comm,
  1612	            edge_comm=_dummy_comm,
  1613	            vertex_comm=_dummy_comm,
  1614	        )
  1615	        lm = build_local_mesh(gm_np, part)
  1616	        partitions.append(part)
  1617	        local_meshes_raw.append(lm)
  1618	
  1619	    # Uniform padding to the maximum local sizes across all devices.
  1620	    max_lc = max(p.n_local_cells for p in partitions)
  1621	    max_le = max(p.n_local_edges for p in partitions)
  1622	    max_lv = max(p.n_local_vertices for p in partitions)
  1623	
  1624	    local_meshes = [
  1625	        _pad_local_mesh_to(lm, max_lc, max_le, max_lv)
  1626	        for lm in local_meshes_raw
  1627	    ]
  1628	
  1629	    # Stack into a single pytree with a leading device dimension.
  1630	    stacked_meshes = jax.tree.map(
  1631	        lambda *leaves: jnp.stack(leaves, axis=0),
  1632	        *local_meshes,
  1633	    )
  1634	
  1635	    # Gather indices: for each device, global cell/edge indices of its
  1636	    # local entities (owned + halo), padded with 0 for ghost slots.
  1637	    gather_cells = np.zeros((n_dev, max_lc), dtype=np.int64)
  1638	    gather_edges = np.zeros((n_dev, max_le), dtype=np.int64)
  1639	    n_owned_cells = []
  1640	    n_owned_edges = []
  1641	    for rank, part in enumerate(partitions):
  1642	        gather_cells[rank, : part.n_local_cells] = part.local_cells
  1643	        gather_edges[rank, : part.n_local_edges] = part.local_edges
  1644	        n_owned_cells.append(part.n_owned_cells)
  1645	        n_owned_edges.append(part.n_owned_edges)
  1646	
  1647	    return (
  1648	        stacked_meshes,
  1649	        jnp.array(gather_cells),
  1650	        jnp.array(gather_edges),
  1651	        n_owned_cells,
  1652	        n_owned_edges,
  1653	        max_lc,
  1654	        max_le,
  1655	        partitions,
  1656	        cell_owner,
  1657	    )
  1658	
  1659	
  1660	def _greedy_edge_coloring_ordered(comm_pairs, order):
  1661	    """First-fit edge coloring visiting ``order`` (a list of normalized
  1662	    ``(min,max)`` pairs). Always a PROPER coloring; the color count depends
  1663	    on the visitation order.
  1664	    """
  1665	    from collections import defaultdict
packages/core/legoesm/parallel/sharded_dynamics.py-700-        Per-face resolution — logging/prewarm metadata forwarded to
packages/core/legoesm/parallel/sharded_dynamics.py-701-        ``activate_spmd_halo_backend``.  The old ppermute-vs-all_gather
packages/core/legoesm/parallel/sharded_dynamics.py-702-        volume auto-selection is RETIRED: ppermute is always selected;
packages/core/legoesm/parallel/sharded_dynamics.py:703:        all_gather only via explicit ``LEGOESM_SPMD_FORCE_ALLGATHER=1``.
packages/core/legoesm/parallel/sharded_dynamics.py-704-    nlev : int, optional
packages/core/legoesm/parallel/sharded_dynamics.py-705-        Number of vertical levels (logging only, see ``n``).
packages/core/legoesm/parallel/sharded_dynamics.py-706-
--
packages/core/legoesm/parallel/sharded_dynamics.py-720-    cross-face reads — the pre-activation behavior this Notes section
packages/core/legoesm/parallel/sharded_dynamics.py-721-    used to describe — replicates ALL compute per device (HLO probe job
packages/core/legoesm/parallel/sharded_dynamics.py-722-    8456476); the all_gather kernels survive only as the explicit
packages/core/legoesm/parallel/sharded_dynamics.py:723:    ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` diagnostic.
packages/core/legoesm/parallel/sharded_dynamics.py-724-
packages/core/legoesm/parallel/sharded_dynamics.py-725-    For sub-face tiling (>6 devices), additional tile-boundary
packages/core/legoesm/parallel/sharded_dynamics.py-726-    exchange is needed; the SPMD halo backend is NOT activated there
--
packages/core/legoesm/parallel/sharded_dynamics.py-741-    # for every face-sharded count and at halo=2 — the allgather
packages/core/legoesm/parallel/sharded_dynamics.py-742-    # variant provably replicated ALL compute per device (HLO probe job
packages/core/legoesm/parallel/sharded_dynamics.py-743-    # 8456476, per-device FLOPs ratio 1.00 at 2 devices) and is now an
packages/core/legoesm/parallel/sharded_dynamics.py:744:    # explicit diagnostic opt-in only (LEGOESM_SPMD_FORCE_ALLGATHER=1).
packages/core/legoesm/parallel/sharded_dynamics.py-745-    _n = config.n_devices
packages/core/legoesm/parallel/sharded_dynamics.py-746-    _tiling = getattr(config, 'tiling', (1, 1))
packages/core/legoesm/parallel/sharded_dynamics.py-747-    _face_ok = (_n in (1, 2, 3, 6) and _tiling == (1, 1))
--
packages/core/legoesm/parallel/sharded_dynamics.py-1244-#: consumed by both the production step factory and ``spmd_schedule_cost``:
packages/core/legoesm/parallel/sharded_dynamics.py-1245-#: a score computed at a different depth describes a different comm graph, and
packages/core/legoesm/parallel/sharded_dynamics.py-1246-#: two independently hardcoded 3s let production drift unnoticed.
packages/core/legoesm/parallel/sharded_dynamics.py:1247:SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py-1248-
packages/core/legoesm/parallel/sharded_dynamics.py-1249-
packages/core/legoesm/parallel/sharded_dynamics.py-1250-def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
packages/core/legoesm/parallel/sharded_dynamics.py:1251:                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py:1252:                       ppermute_cells_per_device_threshold=2_000):
packages/core/legoesm/parallel/sharded_dynamics.py-1253-    """How much halo communication one ownership choice costs, computed offline.
packages/core/legoesm/parallel/sharded_dynamics.py-1254-
packages/core/legoesm/parallel/sharded_dynamics.py-1255-    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
--
packages/core/legoesm/parallel/sharded_dynamics.py-1259-    allocation.
packages/core/legoesm/parallel/sharded_dynamics.py-1260-
packages/core/legoesm/parallel/sharded_dynamics.py-1261-    It calls the SAME builders production calls
packages/core/legoesm/parallel/sharded_dynamics.py:1262:    (:func:`_build_voronoi_partition_infra` then
packages/core/legoesm/parallel/sharded_dynamics.py:1263:    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
packages/core/legoesm/parallel/sharded_dynamics.py-1264-    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
packages/core/legoesm/parallel/sharded_dynamics.py-1265-    rounds where the real depth-3-plus-closure graph reports 12-14.
packages/core/legoesm/parallel/sharded_dynamics.py-1266-
--
packages/core/legoesm/parallel/sharded_dynamics.py-1271-      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
packages/core/legoesm/parallel/sharded_dynamics.py-1272-      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
packages/core/legoesm/parallel/sharded_dynamics.py-1273-    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
packages/core/legoesm/parallel/sharded_dynamics.py:1274:      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
packages/core/legoesm/parallel/sharded_dynamics.py-1275-      keeps the best; equality is MEASURED (compare the returned
packages/core/legoesm/parallel/sharded_dynamics.py-1276-      ``max_degree``), never assumed.  Do not claim "the colouring is already
packages/core/legoesm/parallel/sharded_dynamics.py-1277-      optimal so only ownership can help" from this function.
packages/core/legoesm/parallel/sharded_dynamics.py-1278-    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
packages/core/legoesm/parallel/sharded_dynamics.py:1279:      cells/device is below ``ppermute_cells_per_device_threshold``, in which
packages/core/legoesm/parallel/sharded_dynamics.py-1280-      case there is no ppermute schedule and this number is counterfactual --
packages/core/legoesm/parallel/sharded_dynamics.py-1281-      see the returned ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py-1282-
--
packages/core/legoesm/parallel/sharded_dynamics.py-1307-    already_reordered : bool
packages/core/legoesm/parallel/sharded_dynamics.py-1308-    halo_depth : int
packages/core/legoesm/parallel/sharded_dynamics.py-1309-        Must match production (3) or the graph is a different graph.
packages/core/legoesm/parallel/sharded_dynamics.py:1310:    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py-1311-        Mirror of the production auto-select threshold, only used to report
packages/core/legoesm/parallel/sharded_dynamics.py-1312-        ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py-1313-
--
packages/core/legoesm/parallel/sharded_dynamics.py-1366-            f"mesh.")
packages/core/legoesm/parallel/sharded_dynamics.py-1367-    (
packages/core/legoesm/parallel/sharded_dynamics.py-1368-        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
packages/core/legoesm/parallel/sharded_dynamics.py:1369:    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
packages/core/legoesm/parallel/sharded_dynamics.py-1370-    cells_per = n_cells // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-1371-    edges_per = n_edges // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py:1372:    sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py-1373-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
packages/core/legoesm/parallel/sharded_dynamics.py-1374-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1375-    return {
--
packages/core/legoesm/parallel/sharded_dynamics.py-1389-        "max_degree": int(sched.get("max_degree", -1)),
packages/core/legoesm/parallel/sharded_dynamics.py-1390-        "coloring_method": sched["coloring_method"],
packages/core/legoesm/parallel/sharded_dynamics.py-1391-        # Production returns before selecting a strategy at n_dev==1, and a
packages/core/legoesm/parallel/sharded_dynamics.py:1392:        # caller may force halo_strategy; this reports what AUTO would pick.
packages/core/legoesm/parallel/sharded_dynamics.py-1393-        "production_strategy": (
packages/core/legoesm/parallel/sharded_dynamics.py-1394-            None if n_dev == 1 else
packages/core/legoesm/parallel/sharded_dynamics.py:1395:            ("allgather" if cells_per < ppermute_cells_per_device_threshold
packages/core/legoesm/parallel/sharded_dynamics.py-1396-             else "ppermute")),
packages/core/legoesm/parallel/sharded_dynamics.py-1397-        "cells_per_device": cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py-1398-        "max_local_cells": int(max_lc),
--
packages/core/legoesm/parallel/sharded_dynamics.py-1400-    }
packages/core/legoesm/parallel/sharded_dynamics.py-1401-
packages/core/legoesm/parallel/sharded_dynamics.py-1402-
packages/core/legoesm/parallel/sharded_dynamics.py:1403:def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
packages/core/legoesm/parallel/sharded_dynamics.py-1404-    """Pre-compute per-device local meshes and gather/scatter indices.
packages/core/legoesm/parallel/sharded_dynamics.py-1405-
packages/core/legoesm/parallel/sharded_dynamics.py-1406-    After ``reorder_voronoi_for_sharding`` the global mesh has *both*
--
packages/core/legoesm/parallel/sharded_dynamics.py-1759-    return best_colors, max_degree
packages/core/legoesm/parallel/sharded_dynamics.py-1760-
packages/core/legoesm/parallel/sharded_dynamics.py-1761-
packages/core/legoesm/parallel/sharded_dynamics.py:1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py-1763-                             edges_per, max_lc, max_le):
packages/core/legoesm/parallel/sharded_dynamics.py-1764-    """Build a ppermute-based halo exchange schedule.
packages/core/legoesm/parallel/sharded_dynamics.py-1765-
--
packages/core/legoesm/parallel/sharded_dynamics.py-2109-    return cell_local[:max_lc], u_local[:max_le]
packages/core/legoesm/parallel/sharded_dynamics.py-2110-
packages/core/legoesm/parallel/sharded_dynamics.py-2111-
packages/core/legoesm/parallel/sharded_dynamics.py:2112:def make_voronoi_sharded_step(
packages/core/legoesm/parallel/sharded_dynamics.py-2113-    model,
packages/core/legoesm/parallel/sharded_dynamics.py-2114-    dev_config: DeviceConfig,
packages/core/legoesm/parallel/sharded_dynamics.py-2115-    *,
packages/core/legoesm/parallel/sharded_dynamics.py:2116:    halo_strategy: str = "auto",
packages/core/legoesm/parallel/sharded_dynamics.py:2117:    ppermute_cells_per_device_threshold: int = 2_000,
packages/core/legoesm/parallel/sharded_dynamics.py-2118-    return_phys_state: bool = False,
packages/core/legoesm/parallel/sharded_dynamics.py-2119-):
packages/core/legoesm/parallel/sharded_dynamics.py-2120-    """Create a halo-partitioned multi-GPU step for Voronoi (MPAS/TRiSK) grids.
--
packages/core/legoesm/parallel/sharded_dynamics.py-2158-        ``.config``.
packages/core/legoesm/parallel/sharded_dynamics.py-2159-    dev_config : DeviceConfig
1092-# ============================================================================
1093-#
1094-# create_voronoi_mesh runs Lloyd relaxation (scipy SphericalVoronoi) which is
1095:# DETERMINISTIC in (subdivision_level, radius, lloyd_iterations, omega) when
1096-# density_fn is None, and slow (~10^2 s at level>=6).  Under MPI every rank
1097-# rebuilds the SAME mesh; at 32 ranks/node the redundant builds contend for
1098-# cores and exceed the job walltime (the multi-node "hang" was this, not an MPI
--
1140-
1141-
1142-def _voronoi_cache_path(
1143:    subdivision_level: int, radius: float, lloyd_iterations: int, omega: float
1144-) -> str:
1145-    """Deterministic cache filename keyed on every parameter that changes the mesh.
1146-
--
1151-    x64 = bool(jax.config.read("jax_enable_x64"))
1152-    name = (
1153-        f"scvt_v{_MESH_CACHE_VERSION}_lvl{subdivision_level}"
1154:        f"_r{radius!r}_omega{omega!r}_lloyd{lloyd_iterations}"
1155-        f"_x64{int(x64)}.npz"
1156-    )
1157-    return os.path.join(_voronoi_cache_dir(), name)
--
1221-def prewarm_voronoi_cache(
1222-    subdivision_level: int,
1223-    radius: float = constants.R_earth,
1224:    lloyd_iterations: int = 50,
1225-    omega: float = constants.Omega,
1226-) -> str:
1227-    """Build (if absent) and cache the uniform SCVT mesh; return the cache path.
--
1233-    if _voronoi_cache_disabled():
1234-        raise RuntimeError(
1235-            f"{_MESH_CACHE_DISABLE_ENV} is set; cannot pre-warm a disabled cache.")
1236:    path = _voronoi_cache_path(subdivision_level, radius, lloyd_iterations, omega)
1237:    create_voronoi_mesh(
1238-        subdivision_level, radius=radius,
1239:        lloyd_iterations=lloyd_iterations, omega=omega)
1240-    return path
1241-
1242-
1243:def create_voronoi_mesh(
1244-    subdivision_level: int,
1245-    radius: float = constants.R_earth,
1246:    lloyd_iterations: int = 50,
1247-    omega: float = constants.Omega,
1248-    density_fn=None,
1249-) -> VoronoiMesh:
--
1260-        level=3: 642 cells, level=4: 2562, level=5: 10242.
1261-    radius : float
1262-        Sphere radius [m]. Default: Earth radius.
1263:    lloyd_iterations : int
1264-        Number of Lloyd relaxation iterations. Default: 50.
1265-    omega : float
1266-        Rotation rate [rad/s]. Default: Earth rotation.
--
1275-        the uniform quasi-uniform SCVT (unchanged).  Host-side mesh generation —
1276-        the callable is plain NumPy, never traced.  NOTE: Lloyd relaxation converges
1277-        linearly, so the achieved refinement contrast grows with
1278:        ``lloyd_iterations``; for strong/precise variable resolution prefer a
1279-        JIGSAW-built mesh loaded via :func:`load_mpas_mesh`.
1280-
1281-    Returns
--
1305-    cache_path = None
1306-    if use_cache:
1307-        cache_path = _voronoi_cache_path(
1308:            subdivision_level, radius, lloyd_iterations, omega)
1309-        cached = _load_voronoi_cache(cache_path)
1310-        if cached is not None:
1311-            return cached
--
1363-        verts, triangles = _bisect_mesh(verts, triangles, subdivision_level)
1364-
1365-    # Step 3: Lloyd relaxation for SCVT (density-weighted when density_fn given)
1366:    if lloyd_iterations > 0 and subdivision_level > 0:
1367-        cell_points = _lloyd_relaxation(
1368:            verts, n_iter=lloyd_iterations, density_fn=density_fn)
1369-    elif density_fn is not None:
1370-        raise ValueError(
1371:            "density_fn requires lloyd_iterations > 0 and subdivision_level > 0 "
1372-            "(the variable-resolution mesh is produced by density-weighted Lloyd "
1373-            "relaxation; with no relaxation the icosahedral seed stays uniform)."
1374-        )
./tests/bench/test_bench_voronoi_partition_methods.py:216:    ref = spmd_schedule_cost(mesh, 2, method="sfc")
./scripts/bench/bench_voronoi_partition_methods.py:217:    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
./tests/parallel/test_spmd_schedule_cost.py:22:    c = sd.spmd_schedule_cost(mesh, 4)
./tests/parallel/test_spmd_schedule_cost.py:40:    c = sd.spmd_schedule_cost(mesh, 8)
./tests/parallel/test_spmd_schedule_cost.py:46:    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
./tests/parallel/test_spmd_schedule_cost.py:48:        sd.spmd_schedule_cost(mesh, 0)
./tests/parallel/test_spmd_schedule_cost.py:54:    c = sd.spmd_schedule_cost(mesh, 8,
./tests/parallel/test_spmd_schedule_cost.py:57:    big = sd.spmd_schedule_cost(mesh, 8,
./tests/parallel/test_spmd_schedule_cost.py:67:    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
./tests/parallel/test_spmd_schedule_cost.py:68:    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
./tests/parallel/test_spmd_schedule_cost.py:77:        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
./tests/parallel/test_spmd_schedule_cost.py:85:    same = sd.spmd_schedule_cost(mesh, 4, method="sfc")
./tests/parallel/test_spmd_schedule_cost.py:86:    split_for_16 = sd.spmd_schedule_cost(mesh, 4, method="sfc",
./tests/parallel/test_spmd_schedule_cost.py:97:    a = sd.spmd_schedule_cost(mesh, 8, method="sfc")
./tests/parallel/test_spmd_schedule_cost.py:98:    b = sd.spmd_schedule_cost(mesh, 8, method="sfc")
./tests/parallel/test_spmd_schedule_cost.py:115:    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
./tests/parallel/test_spmd_schedule_cost.py:125:        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")
./tests/parallel/test_spmd_schedule_cost.py:138:    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
./tests/parallel/test_spmd_schedule_cost.py:147:        sd.spmd_schedule_cost(mesh, 4, method="sfc", reorder_target=3)
./tests/parallel/test_spmd_schedule_cost.py:152:        sd.spmd_schedule_cost(mesh, 3.9)
./tests/parallel/test_spmd_schedule_cost.py:169:    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None
./packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '2100,2460p'; nl -ba tests/parallel/test_spmd_schedule_cost.py | sed -n '1,240p'; rg -n -C 3 \"n_rounds|n_rounds_greedy|max_degree|_multi_ordering_edge_coloring|_build_ppermute_schedule\" tests packages/core/legoesm -g '*.py' | head -n 300; rg -n -C 3 \"--lloyd|lloyd_iterations|halo-depth|schedule-cost|metadata\" tests/bench/test_bench_voronoi_partition_methods.py scripts/bench/bench_voronoi_partition_methods.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 2 in 756ms:
  2100	        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
  2101	        recv_packed = jax.lax.ppermute(
  2102	            send_packed, "device", perm=ppermute_perms[r])
  2103	        split_at = send_c_flat.shape[0]       # static
  2104	        recv_c = recv_packed[:split_at].reshape(send_c.shape)
  2105	        recv_e = recv_packed[split_at:].reshape(send_e.shape)
  2106	        cell_local = cell_local.at[rc[0]].set(recv_c)
  2107	        u_local = u_local.at[re[0]].set(recv_e)
  2108	
  2109	    return cell_local[:max_lc], u_local[:max_le]
  2110	
  2111	
  2112	def make_voronoi_sharded_step(
  2113	    model,
  2114	    dev_config: DeviceConfig,
  2115	    *,
  2116	    halo_strategy: str = "auto",
  2117	    ppermute_cells_per_device_threshold: int = 2_000,
  2118	    return_phys_state: bool = False,
  2119	):
  2120	    """Create a halo-partitioned multi-GPU step for Voronoi (MPAS/TRiSK) grids.
  2121	
  2122	    Instead of replicating the full state and redundantly computing the
  2123	    full step on every device, this implementation:
  2124	
  2125	    1. Pre-computes per-device local meshes (owned cells/edges + halo)
  2126	       at setup time via domain decomposition.
  2127	    2. At each RK stage (``config.time_integrator`` via
  2128	       ``dispatch_integrator`` — same integrator code as the serial
  2129	       ``_step_jit``), exchanges only halo data between neighboring
  2130	       devices (not the full state), then computes tendencies on the
  2131	       local mesh.  The packed exchange carries the FULL prognostic
  2132	       state: u (edge) plus T, p_s, phis and every tracer (cell) in one
  2133	       flat ppermute payload per neighbor round.
  2134	    3. Applies operator-split physics ONCE on the post-dynamics state
  2135	       (traced ``forcing`` + prognostic ``phys_state`` carry threaded
  2136	       through), then the temperature/tracer floors and the global mass
  2137	       fix — mirroring the serial ``MPASPrimitiveEquationModel._step_jit``
  2138	       operator ordering exactly.
  2139	
  2140	    Local-only metadata: the per-device local meshes (stacked with a
  2141	    leading device axis), the ppermute schedule index arrays, and the
  2142	    mass-fix ``areaCell`` are ``P("device")``-sharded and passed as
  2143	    ARGUMENTS into the jitted step (multi-controller-safe: sharded jit
  2144	    args are legal where sharded closure constants raise at trace time)
  2145	    — each device holds ONLY its own local mesh + schedule rows, never
  2146	    the global connectivity.  The one remaining NON-local metadata is
  2147	    the global-mesh closure handed to the operator-split physics term:
  2148	    column-local physics runs OUTSIDE shard_map on the GSPMD-sharded
  2149	    global arrays, reading only replicated 1-D cell fields (latCell
  2150	    etc. — O(nCells) scalars, not the 2-D connectivity).  See the
  2151	    physics block below and
  2152	    ``docs/performance/scaling/mpas_atm_native_step_audit.md``.
  2153	
  2154	    Parameters
  2155	    ----------
  2156	    model
  2157	        ``MPASPrimitiveEquationModel`` with ``.mesh``, ``.sigma_coord``,
  2158	        ``.config``.
  2159	    dev_config : DeviceConfig
  2160	        From :func:`~legoesm.parallel.mesh.create_voronoi_device_mesh`.
  2161	    halo_strategy : str
  2162	        ``"auto"`` (default) selects ``"ppermute"`` for large grids and
  2163	        ``"allgather"`` for small ones based on
  2164	        *ppermute_cells_per_device_threshold*.
  2165	        ``"ppermute"`` forces neighbor-only exchange via
  2166	        ``jax.lax.ppermute`` — O(halo) communication.
  2167	        ``"allgather"`` forces the full-state all-gather —
  2168	        O(N) communication.
  2169	    ppermute_cells_per_device_threshold : int
  2170	        When ``halo_strategy="auto"``, use ppermute only if each device
  2171	        owns at least this many cells.  Below this threshold the
  2172	        per-round packing/scatter overhead of ppermute exceeds the
  2173	        communication savings over allgather.  Default: 2 000.
  2174	        (Lowered from 25 000 to avoid the O(N) allgather bottleneck
  2175	        on moderate icosahedral grids like I5 with 2–4 GPUs.)
  2176	    return_phys_state : bool
  2177	        ``False`` (default, backward-compatible): the returned step is
  2178	        ``step(state, dt, physics_fn=None, forcing=None, phys_state=None)
  2179	        -> state`` — the physics carry is dropped, so a STATEFUL
  2180	        physics_fn is refused loudly (issue #405/#413).  ``True``: the
  2181	        step returns ``(state, phys_state_out)`` — full operator-split
  2182	        production parity with ``make_voronoi_mpi_step(
  2183	        return_phys_state=True)``; the prognostic physics carry (TKE /
  2184	        convection state) and the traced per-step ``forcing`` (e.g.
  2185	        prescribed ``T_sfc``) are threaded through.
  2186	
  2187	    Returns
  2188	    -------
  2189	    callable
  2190	        ``step(state, dt, physics_fn=None, forcing=None, phys_state=None)``
  2191	        returning ``state`` (``return_phys_state=False``) or
  2192	        ``(state, phys_state_out)`` (``return_phys_state=True``).
  2193	        ``physics_fn`` follows the MPAS operator-split convention
  2194	        (``physics_fn(state, mesh, sigma_coord, *, phys_state, forcing)``
  2195	        returning ``MPASHydrostaticTendencies`` or a ``(tendencies,
  2196	        phys_state_out)`` tuple) and is captured in the jitted closure,
  2197	        never traced as an argument (same convention as
  2198	        :class:`CompiledShardedStep`).  Each distinct ``physics_fn``
  2199	        identity compiles a separate executable; ``physics_fn=None``
  2200	        compiles exactly the dynamics-only graph.  ``forcing`` /
  2201	        ``phys_state`` are jit arguments (NOT static) so new values each
  2202	        step do not retrace (SegmentForcing doctrine); their pytree
  2203	        STRUCTURE must stay stable across steps.  On a single-device
  2204	        config this returns ``model.step``, whose signature is
  2205	        call-compatible (state-only contract).
  2206	    """
  2207	    if dev_config.n_devices <= 1 or dev_config.mesh is None:
  2208	        if return_phys_state:
  2209	            # model.step returns only the state and stashes the carry on
  2210	            # the model EAGERLY (skipped under an outer trace, gh-417) —
  2211	            # returning it here would silently drop/reseed the carry
  2212	            # inside scan-driven callers.  Refuse loudly; the serial
  2213	            # carry contract is the model/driver's own.
  2214	            raise ValueError(
  2215	                "make_voronoi_sharded_step(return_phys_state=True) needs "
  2216	                "a multi-device config; on a single device use "
  2217	                "model.step (eager, carry stashed on the model) or the "
  2218	                "ModelDriver loop, which threads the carry."
  2219	            )
  2220	        return model.step
  2221	
  2222	    from legoesm.core.precision import cast_pytree
  2223	    from legoesm.core.state import MPASHydrostaticState
  2224	    from legoesm.parallel.mesh import multiprocess_safe_device_put
  2225	    from legoesm.parallel.shard_map_compat import shard_map
  2226	    from legoesm.timestepping.dispatch import dispatch_integrator
  2227	    from legoesm.timestepping.integration import (
  2228	        refuse_unthreaded_stateful_physics,
  2229	    )
  2230	
  2231	    n_dev = dev_config.n_devices
  2232	    voronoi_dims = dev_config.voronoi_dims
  2233	    if voronoi_dims is None:
  2234	        raise ValueError("dev_config.voronoi_dims must be set for Voronoi grids")
  2235	    nCells, nEdges, _nVerts = voronoi_dims
  2236	    jax_mesh = dev_config.mesh
  2237	
  2238	    cells_per = nCells // n_dev
  2239	    edges_per = nEdges // n_dev
  2240	
  2241	    global_mesh = model.mesh
  2242	    sigma = model.sigma_coord
  2243	    cfg = model.config
  2244	
  2245	    # The MPAS RHS comes from the passed-in model instance, not an atmosphere
  2246	    # import — this substrate ``parallel`` module must not depend UP on the
  2247	    # atmosphere component (federation: legoesm-core stays standalone-installable).
  2248	    # The free function is needed (not ``.tendencies``) because each device runs
  2249	    # it on its own rank-local, traced mesh.
  2250	    mpas_hydrostatic_tendencies = getattr(model, "sharded_tendency_fn", None)
  2251	    if mpas_hydrostatic_tendencies is None:
  2252	        raise TypeError(
  2253	            f"{type(model).__name__} does not expose a 'sharded_tendency_fn' "
  2254	            f"staticmethod; the multi-device Voronoi sharder needs the free "
  2255	            f"tendency RHS (state, mesh, sigma_coord, config, *, dt=...) to run "
  2256	            f"on a rank-local mesh without importing the dycore's component."
  2257	        )
  2258	
  2259	    # ------------------------------------------------------------------
  2260	    # Auto-select halo strategy based on grid size per device
  2261	    # ------------------------------------------------------------------
  2262	    if halo_strategy == "auto":
  2263	        if cells_per < ppermute_cells_per_device_threshold:
  2264	            halo_strategy = "allgather"
  2265	            logger.info(
  2266	                "Auto-selected allgather strategy: cells_per_device=%d < "
  2267	                "threshold=%d — ppermute packing overhead would dominate.",
  2268	                cells_per, ppermute_cells_per_device_threshold,
  2269	            )
  2270	        else:
  2271	            halo_strategy = "ppermute"
  2272	            logger.info(
  2273	                "Auto-selected ppermute strategy: cells_per_device=%d >= "
  2274	                "threshold=%d.",
  2275	                cells_per, ppermute_cells_per_device_threshold,
  2276	            )
  2277	
  2278	    # ------------------------------------------------------------------
  2279	    # Setup: build per-device local meshes and gather indices
  2280	    # ------------------------------------------------------------------
  2281	    logger.info(
  2282	        "Building halo-partitioned infrastructure for %d device(s) "
  2283	        "(nCells=%d, nEdges=%d, halo_depth=3, strategy=%s) ...",
  2284	        n_dev, nCells, nEdges, halo_strategy,
  2285	    )
  2286	    t0 = time.time()
  2287	    (
  2288	        stacked_meshes,   # VoronoiMesh pytree with (n_dev, max_l*) leaves
  2289	        gather_cells,     # (n_dev, max_lc)
  2290	        gather_edges,     # (n_dev, max_le)
  2291	        _n_owned_cells,
  2292	        _n_owned_edges,
  2293	        max_lc,
  2294	        max_le,
  2295	        partitions_out,   # list[VoronoiPartition] (for ppermute schedule)
  2296	        cell_owner_out,   # np.ndarray (nCells,) cell ownership
  2297	    ) = _build_voronoi_partition_infra(global_mesh, n_dev,
  2298	                                       halo_depth=SPMD_HALO_DEPTH)
  2299	    logger.info(
  2300	        "  partition setup done in %.2fs  "
  2301	        "(max_local_cells=%d, max_local_edges=%d, cells_per=%d, edges_per=%d)",
  2302	        time.time() - t0, max_lc, max_le, cells_per, edges_per,
  2303	    )
  2304	
  2305	    # LOCAL-ONLY metadata: shard the stacked local meshes on the leading
  2306	    # device axis — device i holds ONLY its own local mesh (leaf slice
  2307	    # [i]), never the other devices' connectivity.  The mesh rides into
  2308	    # the jitted step as an ARGUMENT with P("device") shard_map in_specs
  2309	    # (multi-controller-safe: a sharded jit ARG is legal where a sharded
  2310	    # CLOSURE constant raises at trace time under jax.distributed;
  2311	    # ``multiprocess_safe_device_put`` builds the global array from each
  2312	    # process's local copy).  All leaves are arrays after the jnp.stack
  2313	    # in _build_voronoi_partition_infra (ints become (n_dev,) arrays).
  2314	    dev_sharding = dev_config.face_sharding  # P("device") on axis 0
  2315	    stacked_meshes = jax.tree.map(
  2316	        lambda x: multiprocess_safe_device_put(x, dev_sharding),
  2317	        stacked_meshes,
  2318	    )
  2319	
  2320	    nlev = model.sigma_coord.n_levels
  2321	
  2322	    # ------------------------------------------------------------------
  2323	    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
  2324	    # ------------------------------------------------------------------
  2325	
  2326	    use_ppermute = halo_strategy == "ppermute"
  2327	
  2328	    if use_ppermute:
  2329	        # Build ppermute schedule: neighbor-only halo exchange
  2330	        t1 = time.time()
  2331	        pp_sched = _build_ppermute_schedule(
  2332	            partitions_out, cell_owner_out, n_dev,
  2333	            cells_per, edges_per, max_lc, max_le,
  2334	        )
  2335	        n_rounds = pp_sched['n_rounds']
  2336	        ppermute_perms = pp_sched['ppermute_perms']
  2337	
  2338	        # LOCAL-ONLY metadata: shard the per-round index arrays on the
  2339	        # leading device axis (each device holds only its own schedule
  2340	        # rows) and thread them as shard_map ARGUMENTS — see the stacked
  2341	        # meshes above for why args, not closures.
  2342	        halo_args = tuple(
  2343	            (
  2344	                multiprocess_safe_device_put(
  2345	                    pp_sched['send_cell_idx'][r], dev_sharding),
  2346	                multiprocess_safe_device_put(
  2347	                    pp_sched['recv_cell_pos'][r], dev_sharding),
  2348	                multiprocess_safe_device_put(
  2349	                    pp_sched['send_edge_idx'][r], dev_sharding),
  2350	                multiprocess_safe_device_put(
  2351	                    pp_sched['recv_edge_pos'][r], dev_sharding),
  2352	            )
  2353	            for r in range(n_rounds)
  2354	        )
  2355	
  2356	        # Log halo exchange statistics (dry-state estimate: tracers add
  2357	        # nlev*n_tracers further cell channels to both strategies).
  2358	        total_pp_bytes = sum(
  2359	            hc * (nlev + 2) + he * nlev
  2360	            for hc, he in zip(pp_sched['halo_cells_per_round'],
  2361	                              pp_sched['halo_edges_per_round'])
  2362	        ) * 4  # float32
  2363	        ag_bytes = (nCells * (nlev + 2) + nEdges * nlev) * 4
  2364	        logger.info(
  2365	            "  ppermute schedule: %d rounds, max halo cells/edges per round: %s / %s",
  2366	            n_rounds,
  2367	            pp_sched['halo_cells_per_round'],
  2368	            pp_sched['halo_edges_per_round'],
  2369	        )
  2370	        logger.info(
  2371	            "  comm volume per stage: ppermute ~%.1f KB vs allgather ~%.1f KB (%.1fx reduction)",
  2372	            total_pp_bytes / 1024,
  2373	            ag_bytes / 1024,
  2374	            ag_bytes / max(total_pp_bytes, 1),
  2375	        )
  2376	        logger.info("  ppermute schedule built in %.3fs", time.time() - t1)
  2377	
  2378	    else:
  2379	        # ---- Legacy all-gather strategy: local gather indices ----
  2380	        halo_args = (
  2381	            multiprocess_safe_device_put(gather_cells, dev_sharding),
  2382	            multiprocess_safe_device_put(gather_edges, dev_sharding),
  2383	        )
  2384	
  2385	    # ------------------------------------------------------------------
  2386	    # shard_map kernel: packed full-state halo fill → local tendency
  2387	    # ------------------------------------------------------------------
  2388	    # The kernel is built per canonical tracer-key tuple (the keys are
  2389	    # part of the traced program: cell-pack width and the tracer dict
  2390	    # rebuilt on the local mesh).  Memoized so a stable state structure
  2391	    # reuses one shard_map object → one jit executable (no retrace).
  2392	
  2393	    mesh_in_specs = jax.tree.map(lambda _: P("device"), stacked_meshes)
  2394	    halo_in_specs = jax.tree.map(lambda _: P("device"), halo_args)
  2395	
  2396	    def _make_local_tendency(tkeys: tuple):
  2397	
  2398	        def _local_tendency(u_shard, T_shard, ps_shard, phis_shard,
  2399	                            q_shard, dt_val, mesh_sl, halo_sl):
  2400	            """Inside shard_map: full-state halo fill → local tendency.
  2401	
  2402	            ``q_shard`` is the tracer block ``(cells_per, nlev * n_q)``
  2403	            — tracers concatenated on the trailing axis in the canonical
  2404	            sorted-key WIRE order (width 0 for a dry run).  ``mesh_sl``
  2405	            / ``halo_sl`` are this device's P("device") slices of the
  2406	            stacked local meshes and the halo schedule (leading axis 1).
  2407	            """
  2408	            # Pack ALL cell-centred prognostics into a single buffer
  2409	            # (cells_per, nlev + 2 + nlev*n_q) via the shared wire-layout
  2410	            # helper (also driven directly by the sentinel routing test).
  2411	            cell_pack = _pack_cell_state(T_shard, ps_shard, phis_shard,
  2412	                                         q_shard)
  2413	
  2414	            if use_ppermute:
  2415	                cell_local, u_local = _ppermute_halo_fill(
  2416	                    cell_pack, u_shard, halo_sl, ppermute_perms,
  2417	                    max_lc, max_le,
  2418	                )
  2419	            else:
  2420	                cell_full = jax.lax.all_gather(
  2421	                    cell_pack, "device", axis=0, tiled=True)
  2422	                u_full = jax.lax.all_gather(
  2423	                    u_shard, "device", axis=0, tiled=True)
  2424	                gc, ge = halo_sl
  2425	                cell_local = cell_full[gc[0]]
  2426	                u_local = u_full[ge[0]]
  2427	
  2428	            # Unpack cell fields (inverse of the shared pack helper)
  2429	            T_local, ps_local, phis_local, q_local = _unpack_cell_state(
  2430	                cell_local, nlev)
  2431	
  2432	            # This device's local mesh (leading axis is the length-1
  2433	            # device slice of the stacked meshes).
  2434	            my_mesh = jax.tree.map(lambda x: x[0], mesh_sl)
  2435	
  2436	            tracers_local = None
  2437	            if tkeys:
  2438	                tracers_local = {
  2439	                    k: Field(data=q_local[:, i * nlev:(i + 1) * nlev],
  2440	                             name=k, dims=("nCells", "nlev"),
  2441	                             units="kg/kg", staggering="cell")
  2442	                    for i, k in enumerate(tkeys)
  2443	                }
  2444	
  2445	            # Build local state and compute tendency
  2446	            local_state = MPASHydrostaticState(
  2447	                u=Field(data=u_local, name="u",
  2448	                        dims=("nEdges", "nlev"), units="m/s",
  2449	                        long_name="normal velocity", staggering="edge"),
  2450	                T=Field(data=T_local, name="T",
  2451	                        dims=("nCells", "nlev"), units="K",
  2452	                        long_name="temperature", staggering="cell"),
  2453	                p_s=Field(data=ps_local, name="p_s",
  2454	                          dims=("nCells",), units="Pa",
  2455	                          long_name="surface pressure", staggering="cell"),
  2456	                phis=Field(data=phis_local, name="phis",
  2457	                           dims=("nCells",), units="m^2/s^2",
  2458	                           long_name="surface geopotential",
  2459	                           staggering="cell"),
  2460	                tracers=tracers_local,
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
packages/core/legoesm/parallel/sharded_dynamics.py-1260-
packages/core/legoesm/parallel/sharded_dynamics.py-1261-    It calls the SAME builders production calls
packages/core/legoesm/parallel/sharded_dynamics.py-1262-    (:func:`_build_voronoi_partition_infra` then
packages/core/legoesm/parallel/sharded_dynamics.py:1263:    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
packages/core/legoesm/parallel/sharded_dynamics.py-1264-    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
packages/core/legoesm/parallel/sharded_dynamics.py-1265-    rounds where the real depth-3-plus-closure graph reports 12-14.
packages/core/legoesm/parallel/sharded_dynamics.py-1266-
packages/core/legoesm/parallel/sharded_dynamics.py-1267-    WHAT THE NUMBER IS NOT
packages/core/legoesm/parallel/sharded_dynamics.py-1268-    ----------------------
packages/core/legoesm/parallel/sharded_dynamics.py:1269:    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
packages/core/legoesm/parallel/sharded_dynamics.py:1270:      ``n_rounds`` x (tendency evaluations per step), which depends on the
packages/core/legoesm/parallel/sharded_dynamics.py-1271-      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
packages/core/legoesm/parallel/sharded_dynamics.py-1272-      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
packages/core/legoesm/parallel/sharded_dynamics.py:1273:    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
packages/core/legoesm/parallel/sharded_dynamics.py:1274:      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
packages/core/legoesm/parallel/sharded_dynamics.py-1275-      keeps the best; equality is MEASURED (compare the returned
packages/core/legoesm/parallel/sharded_dynamics.py:1276:      ``max_degree``), never assumed.  Do not claim "the colouring is already
packages/core/legoesm/parallel/sharded_dynamics.py-1277-      optimal so only ownership can help" from this function.
packages/core/legoesm/parallel/sharded_dynamics.py-1278-    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
packages/core/legoesm/parallel/sharded_dynamics.py-1279-      cells/device is below ``ppermute_cells_per_device_threshold``, in which
--
packages/core/legoesm/parallel/sharded_dynamics.py-1314-    Returns
packages/core/legoesm/parallel/sharded_dynamics.py-1315-    -------
packages/core/legoesm/parallel/sharded_dynamics.py-1316-    dict
packages/core/legoesm/parallel/sharded_dynamics.py:1317:        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
packages/core/legoesm/parallel/sharded_dynamics.py:1318:        it against), ``n_rounds_greedy``, ``coloring_method``,
packages/core/legoesm/parallel/sharded_dynamics.py-1319-        ``resolved_method`` (concrete, never ``"auto"``),
packages/core/legoesm/parallel/sharded_dynamics.py-1320-        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
packages/core/legoesm/parallel/sharded_dynamics.py-1321-        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1369-    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
packages/core/legoesm/parallel/sharded_dynamics.py-1370-    cells_per = n_cells // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-1371-    edges_per = n_edges // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py:1372:    sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py-1373-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
packages/core/legoesm/parallel/sharded_dynamics.py-1374-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1375-    return {
--
packages/core/legoesm/parallel/sharded_dynamics.py-1384-                            else int(reorder_target))),
packages/core/legoesm/parallel/sharded_dynamics.py-1385-        "already_reordered": bool(already_reordered),
packages/core/legoesm/parallel/sharded_dynamics.py-1386-        "halo_depth": halo_depth,
packages/core/legoesm/parallel/sharded_dynamics.py:1387:        "n_rounds": int(sched["n_rounds"]),
packages/core/legoesm/parallel/sharded_dynamics.py:1388:        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
packages/core/legoesm/parallel/sharded_dynamics.py:1389:        "max_degree": int(sched.get("max_degree", -1)),
packages/core/legoesm/parallel/sharded_dynamics.py-1390-        "coloring_method": sched["coloring_method"],
packages/core/legoesm/parallel/sharded_dynamics.py-1391-        # Production returns before selecting a strategy at n_dev==1, and a
packages/core/legoesm/parallel/sharded_dynamics.py-1392-        # caller may force halo_strategy; this reports what AUTO would pick.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1679-
packages/core/legoesm/parallel/sharded_dynamics.py-1680-def _greedy_edge_coloring(comm_pairs):
packages/core/legoesm/parallel/sharded_dynamics.py-1681-    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
packages/core/legoesm/parallel/sharded_dynamics.py:1682:    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
packages/core/legoesm/parallel/sharded_dynamics.py:1683:    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
packages/core/legoesm/parallel/sharded_dynamics.py-1684-    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
packages/core/legoesm/parallel/sharded_dynamics.py-1685-    pure wall-clock.
packages/core/legoesm/parallel/sharded_dynamics.py-1686-    """
--
packages/core/legoesm/parallel/sharded_dynamics.py-1711-_COLORING_SHUFFLE_SEEDS = tuple(range(16))
packages/core/legoesm/parallel/sharded_dynamics.py-1712-
packages/core/legoesm/parallel/sharded_dynamics.py-1713-
packages/core/legoesm/parallel/sharded_dynamics.py:1714:def _multi_ordering_edge_coloring(comm_pairs):
packages/core/legoesm/parallel/sharded_dynamics.py-1715-    """Proper edge coloring via multi-start first-fit; returns the coloring
packages/core/legoesm/parallel/sharded_dynamics.py-1716-    using the FEWEST colors (= ppermute rounds) across several deterministic
packages/core/legoesm/parallel/sharded_dynamics.py-1717-    visitation orders.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1719-    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
packages/core/legoesm/parallel/sharded_dynamics.py-1720-    the sorted order can overshoot the chromatic index by up to 3 rounds at
packages/core/legoesm/parallel/sharded_dynamics.py-1721-    16 devices, while a degree-descending or shuffled order reaches the
packages/core/legoesm/parallel/sharded_dynamics.py:1722:    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
packages/core/legoesm/parallel/sharded_dynamics.py-1723-    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
packages/core/legoesm/parallel/sharded_dynamics.py-1724-    coloring by construction, so taking the min can NEVER produce an
packages/core/legoesm/parallel/sharded_dynamics.py-1725-    invalid schedule and can never regress below the legacy sorted greedy.
packages/core/legoesm/parallel/sharded_dynamics.py-1726-
packages/core/legoesm/parallel/sharded_dynamics.py-1727-    Deterministic across ranks (sorted + degree orders + fixed-seed
packages/core/legoesm/parallel/sharded_dynamics.py:1728:    shuffles). Returns ``(edge_colors, max_degree)``.
packages/core/legoesm/parallel/sharded_dynamics.py-1729-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1730-    import random
packages/core/legoesm/parallel/sharded_dynamics.py-1731-    from collections import defaultdict
--
packages/core/legoesm/parallel/sharded_dynamics.py-1735-    for u, v in edges:
packages/core/legoesm/parallel/sharded_dynamics.py-1736-        deg[u] += 1
packages/core/legoesm/parallel/sharded_dynamics.py-1737-        deg[v] += 1
packages/core/legoesm/parallel/sharded_dynamics.py:1738:    max_degree = max(deg.values(), default=0)
packages/core/legoesm/parallel/sharded_dynamics.py-1739-
packages/core/legoesm/parallel/sharded_dynamics.py-1740-    orders = [
packages/core/legoesm/parallel/sharded_dynamics.py-1741-        edges,                                                   # sorted
--
packages/core/legoesm/parallel/sharded_dynamics.py-1754-        rounds = max(ec.values(), default=-1) + 1
packages/core/legoesm/parallel/sharded_dynamics.py-1755-        if best_rounds is None or rounds < best_rounds:
packages/core/legoesm/parallel/sharded_dynamics.py-1756-            best_rounds, best_colors = rounds, ec
packages/core/legoesm/parallel/sharded_dynamics.py:1757:            if best_rounds <= max_degree:
packages/core/legoesm/parallel/sharded_dynamics.py-1758-                break            # hit the chromatic-index floor — optimal
packages/core/legoesm/parallel/sharded_dynamics.py:1759:    return best_colors, max_degree
packages/core/legoesm/parallel/sharded_dynamics.py-1760-
packages/core/legoesm/parallel/sharded_dynamics.py-1761-
packages/core/legoesm/parallel/sharded_dynamics.py:1762:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py-1763-                             edges_per, max_lc, max_le):
packages/core/legoesm/parallel/sharded_dynamics.py-1764-    """Build a ppermute-based halo exchange schedule.
packages/core/legoesm/parallel/sharded_dynamics.py-1765-
--
packages/core/legoesm/parallel/sharded_dynamics.py-1780-    Returns
packages/core/legoesm/parallel/sharded_dynamics.py-1781-    -------
packages/core/legoesm/parallel/sharded_dynamics.py-1782-    dict with keys:
packages/core/legoesm/parallel/sharded_dynamics.py:1783:        n_rounds, n_rounds_greedy, max_degree, coloring_method,
packages/core/legoesm/parallel/sharded_dynamics.py-1784-        ppermute_perms, send_cell_idx, recv_cell_pos,
packages/core/legoesm/parallel/sharded_dynamics.py-1785-        send_edge_idx, recv_edge_pos, halo_cells_per_round,
packages/core/legoesm/parallel/sharded_dynamics.py-1786-        halo_edges_per_round.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1823-
packages/core/legoesm/parallel/sharded_dynamics.py-1824-    if not comm_pairs:
packages/core/legoesm/parallel/sharded_dynamics.py-1825-        return {
packages/core/legoesm/parallel/sharded_dynamics.py:1826:            'n_rounds': 0,
packages/core/legoesm/parallel/sharded_dynamics.py:1827:            'n_rounds_greedy': 0,
packages/core/legoesm/parallel/sharded_dynamics.py:1828:            'max_degree': 0,
packages/core/legoesm/parallel/sharded_dynamics.py-1829-            'coloring_method': 'none',
packages/core/legoesm/parallel/sharded_dynamics.py-1830-            'ppermute_perms': [],
packages/core/legoesm/parallel/sharded_dynamics.py-1831-            'send_cell_idx': [],
--
packages/core/legoesm/parallel/sharded_dynamics.py-1841-    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
packages/core/legoesm/parallel/sharded_dynamics.py-1842-    #    fewer colors = directly less wall-clock. First-fit greedy is
packages/core/legoesm/parallel/sharded_dynamics.py-1843-    #    order-sensitive; the multi-start coloring reaches the
packages/core/legoesm/parallel/sharded_dynamics.py:1844:    #    chromatic-index floor (= max_degree) on every probed MPAS config
packages/core/legoesm/parallel/sharded_dynamics.py-1845-    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
packages/core/legoesm/parallel/sharded_dynamics.py-1846-    #    devices). It can never regress: the legacy sorted order is one of
packages/core/legoesm/parallel/sharded_dynamics.py-1847-    #    its candidates and it takes the min. Both are verified proper.
packages/core/legoesm/parallel/sharded_dynamics.py-1848-    # ------------------------------------------------------------------
packages/core/legoesm/parallel/sharded_dynamics.py-1849-    greedy_colors = _greedy_edge_coloring(comm_pairs)
packages/core/legoesm/parallel/sharded_dynamics.py:1850:    n_rounds_greedy = max(greedy_colors.values()) + 1
packages/core/legoesm/parallel/sharded_dynamics.py:1851:    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
packages/core/legoesm/parallel/sharded_dynamics.py:1852:    n_rounds_multi = max(multi_colors.values()) + 1
packages/core/legoesm/parallel/sharded_dynamics.py-1853-    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
packages/core/legoesm/parallel/sharded_dynamics.py-1854-    # on a tie keep the exact legacy sorted-greedy coloring so the produced
packages/core/legoesm/parallel/sharded_dynamics.py-1855-    # schedule is byte-identical to before wherever there is no round win
packages/core/legoesm/parallel/sharded_dynamics.py-1856-    # (the win only appears at high device counts — >=16 on the probed
packages/core/legoesm/parallel/sharded_dynamics.py-1857-    # MPAS meshes). Both colorings are proper.
packages/core/legoesm/parallel/sharded_dynamics.py:1858:    if n_rounds_multi < n_rounds_greedy:
packages/core/legoesm/parallel/sharded_dynamics.py:1859:        edge_colors, n_rounds, coloring_method = (
packages/core/legoesm/parallel/sharded_dynamics.py:1860:            multi_colors, n_rounds_multi, "multi_greedy")
packages/core/legoesm/parallel/sharded_dynamics.py-1861-    else:
packages/core/legoesm/parallel/sharded_dynamics.py:1862:        edge_colors, n_rounds, coloring_method = (
packages/core/legoesm/parallel/sharded_dynamics.py:1863:            greedy_colors, n_rounds_greedy, "greedy")
packages/core/legoesm/parallel/sharded_dynamics.py-1864-    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
packages/core/legoesm/parallel/sharded_dynamics.py-1865-        "improper ppermute edge coloring — two same-round exchanges "
packages/core/legoesm/parallel/sharded_dynamics.py-1866-        "would collide at a device")
--
packages/core/legoesm/parallel/sharded_dynamics.py-1907-    halo_cells_per_round: list[int] = []
packages/core/legoesm/parallel/sharded_dynamics.py-1908-    halo_edges_per_round: list[int] = []
packages/core/legoesm/parallel/sharded_dynamics.py-1909-
packages/core/legoesm/parallel/sharded_dynamics.py:1910:    for r in range(n_rounds):
packages/core/legoesm/parallel/sharded_dynamics.py-1911-        # Max halo size across all pairs in this round
packages/core/legoesm/parallel/sharded_dynamics.py-1912-        max_c = 0
packages/core/legoesm/parallel/sharded_dynamics.py-1913-        max_e = 0
--
packages/core/legoesm/parallel/sharded_dynamics.py-1964-        recv_edge_pos_out.append(jnp.array(re))
packages/core/legoesm/parallel/sharded_dynamics.py-1965-
packages/core/legoesm/parallel/sharded_dynamics.py-1966-    return {
packages/core/legoesm/parallel/sharded_dynamics.py:1967:        'n_rounds': n_rounds,
packages/core/legoesm/parallel/sharded_dynamics.py:1968:        'n_rounds_greedy': n_rounds_greedy,
packages/core/legoesm/parallel/sharded_dynamics.py:1969:        'max_degree': max_degree,
packages/core/legoesm/parallel/sharded_dynamics.py-1970-        'coloring_method': coloring_method,
packages/core/legoesm/parallel/sharded_dynamics.py-1971-        'ppermute_perms': ppermute_perms_out,
packages/core/legoesm/parallel/sharded_dynamics.py-1972-        'send_cell_idx': send_cell_idx_out,
--
packages/core/legoesm/parallel/sharded_dynamics.py-2328-    if use_ppermute:
packages/core/legoesm/parallel/sharded_dynamics.py-2329-        # Build ppermute schedule: neighbor-only halo exchange
packages/core/legoesm/parallel/sharded_dynamics.py-2330-        t1 = time.time()
packages/core/legoesm/parallel/sharded_dynamics.py:2331:        pp_sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py-2332-            partitions_out, cell_owner_out, n_dev,
packages/core/legoesm/parallel/sharded_dynamics.py-2333-            cells_per, edges_per, max_lc, max_le,
packages/core/legoesm/parallel/sharded_dynamics.py-2334-        )
packages/core/legoesm/parallel/sharded_dynamics.py:2335:        n_rounds = pp_sched['n_rounds']
packages/core/legoesm/parallel/sharded_dynamics.py-2336-        ppermute_perms = pp_sched['ppermute_perms']
packages/core/legoesm/parallel/sharded_dynamics.py-2337-
packages/core/legoesm/parallel/sharded_dynamics.py-2338-        # LOCAL-ONLY metadata: shard the per-round index arrays on the
--
packages/core/legoesm/parallel/sharded_dynamics.py-2350-                multiprocess_safe_device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-2351-                    pp_sched['recv_edge_pos'][r], dev_sharding),
packages/core/legoesm/parallel/sharded_dynamics.py-2352-            )
packages/core/legoesm/parallel/sharded_dynamics.py:2353:            for r in range(n_rounds)
packages/core/legoesm/parallel/sharded_dynamics.py-2354-        )
packages/core/legoesm/parallel/sharded_dynamics.py-2355-
packages/core/legoesm/parallel/sharded_dynamics.py-2356-        # Log halo exchange statistics (dry-state estimate: tracers add
--
packages/core/legoesm/parallel/sharded_dynamics.py-2363-        ag_bytes = (nCells * (nlev + 2) + nEdges * nlev) * 4
packages/core/legoesm/parallel/sharded_dynamics.py-2364-        logger.info(
packages/core/legoesm/parallel/sharded_dynamics.py-2365-            "  ppermute schedule: %d rounds, max halo cells/edges per round: %s / %s",
packages/core/legoesm/parallel/sharded_dynamics.py:2366:            n_rounds,
packages/core/legoesm/parallel/sharded_dynamics.py-2367-            pp_sched['halo_cells_per_round'],
packages/core/legoesm/parallel/sharded_dynamics.py-2368-            pp_sched['halo_edges_per_round'],
packages/core/legoesm/parallel/sharded_dynamics.py-2369-        )
--
tests/bench/test_bench_voronoi_partition_methods.py-156-    must be read against."""
tests/bench/test_bench_voronoi_partition_methods.py-157-    mesh = _mesh()
tests/bench/test_bench_voronoi_partition_methods.py-158-    sc = mod.schedule_cost_row(mesh, "geometric", 2)
tests/bench/test_bench_voronoi_partition_methods.py:159:    # max_degree is the graph's own lower bound on a proper edge colouring,
tests/bench/test_bench_voronoi_partition_methods.py-160-    # so a schedule can never beat it.  -1 is the scorer's "not reported"
tests/bench/test_bench_voronoi_partition_methods.py-161-    # sentinel and would make the gap meaningless.
tests/bench/test_bench_voronoi_partition_methods.py:162:    assert sc["max_degree"] >= 1
tests/bench/test_bench_voronoi_partition_methods.py:163:    assert sc["n_rounds"] >= sc["max_degree"]
tests/bench/test_bench_voronoi_partition_methods.py:164:    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
tests/bench/test_bench_voronoi_partition_methods.py:165:    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
tests/bench/test_bench_voronoi_partition_methods.py-166-    assert sc["score_seconds"] >= 0.0
tests/bench/test_bench_voronoi_partition_methods.py-167-
tests/bench/test_bench_voronoi_partition_methods.py-168-
tests/bench/test_bench_voronoi_partition_methods.py-169-@pytest.mark.parametrize(
tests/bench/test_bench_voronoi_partition_methods.py:170:    "n_rounds, max_degree, gap, headroom, proven",
tests/bench/test_bench_voronoi_partition_methods.py-171-    [
tests/bench/test_bench_voronoi_partition_methods.py:172:        # Provably optimal: no proper edge colouring beats max_degree.
tests/bench/test_bench_voronoi_partition_methods.py-173-        (12, 12, 0, 0, True),
tests/bench/test_bench_voronoi_partition_methods.py:174:        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
tests/bench/test_bench_voronoi_partition_methods.py-175-        # buys nothing provable — this is the case that a naive
tests/bench/test_bench_voronoi_partition_methods.py-176-        # "gap > 0 means recolour" rule would over-claim.
tests/bench/test_bench_voronoi_partition_methods.py-177-        (13, 12, 1, 0, False),
--
tests/bench/test_bench_voronoi_partition_methods.py-180-        (14, 10, 4, 3, False),
tests/bench/test_bench_voronoi_partition_methods.py-181-    ])
tests/bench/test_bench_voronoi_partition_methods.py-182-def test_coloring_gap_and_headroom_are_derived_not_assumed(
tests/bench/test_bench_voronoi_partition_methods.py:183:        monkeypatch, n_rounds, max_degree, gap, headroom, proven):
tests/bench/test_bench_voronoi_partition_methods.py-184-    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
tests/bench/test_bench_voronoi_partition_methods.py-185-
tests/bench/test_bench_voronoi_partition_methods.py-186-    Non-vacuity, the hard way: on every mesh small enough to test quickly the
tests/bench/test_bench_voronoi_partition_methods.py-187-    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
tests/bench/test_bench_voronoi_partition_methods.py:188:    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
tests/bench/test_bench_voronoi_partition_methods.py-189-    time), so a real-mesh assertion cannot tell a correct subtraction from a
tests/bench/test_bench_voronoi_partition_methods.py-190-    hardcoded ``0``; that exact mutation passed the first version of this
tests/bench/test_bench_voronoi_partition_methods.py-191-    test.  Stubbing the production scorer with KNOWN values is what makes
tests/bench/test_bench_voronoi_partition_methods.py-192-    the assertion able to fail.
tests/bench/test_bench_voronoi_partition_methods.py-193-
tests/bench/test_bench_voronoi_partition_methods.py-194-    The ``gap == 1`` row is the one that matters: the schedule is a proper
tests/bench/test_bench_voronoi_partition_methods.py:195:    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
tests/bench/test_bench_voronoi_partition_methods.py-196-    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
tests/bench/test_bench_voronoi_partition_methods.py-197-    indistinguishable from optimal (Class 2), and claiming recolouring
tests/bench/test_bench_voronoi_partition_methods.py-198-    headroom there would be an over-claim.
tests/bench/test_bench_voronoi_partition_methods.py-199-    """
tests/bench/test_bench_voronoi_partition_methods.py-200-    stub = {
tests/bench/test_bench_voronoi_partition_methods.py:201:        "n_rounds": n_rounds, "max_degree": max_degree,
tests/bench/test_bench_voronoi_partition_methods.py:202:        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
tests/bench/test_bench_voronoi_partition_methods.py-203-        "resolved_method": "geometric", "halo_depth": 3,
tests/bench/test_bench_voronoi_partition_methods.py-204-        "cells_per_device": 99_999, "production_strategy": "ppermute",
tests/bench/test_bench_voronoi_partition_methods.py-205-    }
--
tests/bench/test_bench_voronoi_partition_methods.py-216-    """Real-mesh sanity on the bound itself: a proper edge colouring can
tests/bench/test_bench_voronoi_partition_methods.py-217-    never use fewer rounds than the graph's max degree, and the multi-start
tests/bench/test_bench_voronoi_partition_methods.py-218-    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
tests/bench/test_bench_voronoi_partition_methods.py:219:    fires, ``max_degree`` is not the degree of the graph being coloured and
tests/bench/test_bench_voronoi_partition_methods.py-220-    every gap-based conclusion built on it is void."""
tests/bench/test_bench_voronoi_partition_methods.py-221-    for method in ("geometric", "sfc"):
tests/bench/test_bench_voronoi_partition_methods.py-222-        for n_ranks in (2, 4, 8):
tests/bench/test_bench_voronoi_partition_methods.py-223-            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
tests/bench/test_bench_voronoi_partition_methods.py:224:            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
tests/bench/test_bench_voronoi_partition_methods.py:225:                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
tests/bench/test_bench_voronoi_partition_methods.py:226:                f"max_degree={sc['max_degree']}")
tests/bench/test_bench_voronoi_partition_methods.py-227-
tests/bench/test_bench_voronoi_partition_methods.py-228-
tests/bench/test_bench_voronoi_partition_methods.py-229-def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
--
tests/bench/test_bench_voronoi_partition_methods.py-245-    mesh = _mesh()
tests/bench/test_bench_voronoi_partition_methods.py-246-    ref = spmd_schedule_cost(mesh, 2, method="sfc")
tests/bench/test_bench_voronoi_partition_methods.py-247-    sc = mod.schedule_cost_row(mesh, "sfc", 2)
tests/bench/test_bench_voronoi_partition_methods.py:248:    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
tests/bench/test_bench_voronoi_partition_methods.py-249-                "coloring_method", "resolved_method", "cells_per_device",
tests/bench/test_bench_voronoi_partition_methods.py-250-                "production_strategy"):
tests/bench/test_bench_voronoi_partition_methods.py-251-        assert sc[key] == ref[key], key
--
tests/bench/test_bench_voronoi_partition_methods.py-271-    assert mod.main() == 0
tests/bench/test_bench_voronoi_partition_methods.py-272-    payload2 = json.loads(out2.read_text())
tests/bench/test_bench_voronoi_partition_methods.py-273-    row = payload2["rows"][0]
tests/bench/test_bench_voronoi_partition_methods.py:274:    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
tests/bench/test_bench_voronoi_partition_methods.py-275-    assert payload2["metadata"]["extra"]["schedule_cost"] is True
tests/bench/test_bench_voronoi_partition_methods.py-276-    # Mesh provenance: a lloyd=0 synthetic mesh must never be readable as a
tests/bench/test_bench_voronoi_partition_methods.py-277-    # production SCVT receipt.
--
packages/core/legoesm/parallel/cubesphere_exchange.py-666-        ``CONNECTIVITY[g][e].reversed`` for the receiving (face, edge)
packages/core/legoesm/parallel/cubesphere_exchange.py-667-        — applied receiver-side to BOTH local and ppermute strips
packages/core/legoesm/parallel/cubesphere_exchange.py-668-        (senders always transmit unreversed source-edge strips).
packages/core/legoesm/parallel/cubesphere_exchange.py:669:    send_lf, send_le : (n_rounds, n_devices, max_slots) int32
packages/core/legoesm/parallel/cubesphere_exchange.py-670-        Strip (local face, edge) this device places in slot ``m`` when
packages/core/legoesm/parallel/cubesphere_exchange.py-671-        it is round ``r``'s sender (garbage rows when idle/padding).
packages/core/legoesm/parallel/cubesphere_exchange.py:672:    recv_tgt : (n_rounds, n_devices, max_slots) int32
packages/core/legoesm/parallel/cubesphere_exchange.py-673-        Flattened ``local_face * 4 + edge`` halo target for slot ``m``,
packages/core/legoesm/parallel/cubesphere_exchange.py-674-        or the sentinel ``4 * k`` (out of bounds → ``mode="drop"``
packages/core/legoesm/parallel/cubesphere_exchange.py-675-        scatter discards it) when the slot is padding or the device
--
packages/core/legoesm/parallel/cubesphere_exchange.py-689-    faces_per_shard: int
packages/core/legoesm/parallel/cubesphere_exchange.py-690-
packages/core/legoesm/parallel/cubesphere_exchange.py-691-
packages/core/legoesm/parallel/cubesphere_exchange.py:692:def _color_device_pairs(pairs, n_rounds):
packages/core/legoesm/parallel/cubesphere_exchange.py-693-    """Exhaustive backtracking edge coloring of directed device pairs.
packages/core/legoesm/parallel/cubesphere_exchange.py-694-
packages/core/legoesm/parallel/cubesphere_exchange.py:695:    Returns a list of ``n_rounds`` lists of (src, dst) pairs where no
packages/core/legoesm/parallel/cubesphere_exchange.py-696-    round repeats a src or a dst (each round is a valid ppermute
packages/core/legoesm/parallel/cubesphere_exchange.py:697:    partial permutation), or ``None`` if no coloring with ``n_rounds``
packages/core/legoesm/parallel/cubesphere_exchange.py-698-    exists.  Deterministic: pairs are processed in sorted order and
packages/core/legoesm/parallel/cubesphere_exchange.py-699-    rounds tried in ascending index.
packages/core/legoesm/parallel/cubesphere_exchange.py-700-    """
packages/core/legoesm/parallel/cubesphere_exchange.py:701:    rounds_src = [set() for _ in range(n_rounds)]
packages/core/legoesm/parallel/cubesphere_exchange.py:702:    rounds_dst = [set() for _ in range(n_rounds)]
packages/core/legoesm/parallel/cubesphere_exchange.py-703-    assignment = [-1] * len(pairs)
packages/core/legoesm/parallel/cubesphere_exchange.py-704-
packages/core/legoesm/parallel/cubesphere_exchange.py-705-    def _bt(i):
rg: unrecognized flag --lloyd|lloyd_iterations|halo-depth|schedule-cost|metadata

exec
/usr/bin/bash -lc "git diff -- scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch; nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '145,310p'; nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '1,380p'; nl -ba tests/parallel/test_spmd_schedule_cost.py | sed -n '1,235p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 420ms:
diff --git a/scripts/bench/bench_voronoi_partition_methods.py b/scripts/bench/bench_voronoi_partition_methods.py
index 6aa0f4a40..fed3f4b0b 100644
--- a/scripts/bench/bench_voronoi_partition_methods.py
+++ b/scripts/bench/bench_voronoi_partition_methods.py
@@ -8,6 +8,52 @@ Scaling-audit item 8.  Two independent measurement layers:
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
+     headroom is exactly ZERO.  Only a partitioner with a new objective
+     (minimize boundary max-degree, not edge cut) can lower the round
+     count.
+   * ``gap == 1`` -> INDISTINGUISHABLE from optimal here.  A Class 2 graph
+     genuinely needs ``Delta + 1``, and deciding Class 1 vs Class 2 is
+     NP-complete, so this does NOT establish that a better colouring
+     exists.  Ceiling either way: one round.
+   * ``gap >= 2`` -> at least ``gap - 1`` rounds of real recolouring
+     headroom (Vizing caps the optimum at ``Delta + 1``).
+
+   Consequence worth stating before any measurement: wherever the
+   multi-start search lands on ``Delta`` or ``Delta + 1`` — which it does
+   on every configuration probed so far — recolouring is capped at ONE
+   round out of 12-14, i.e. <= ~7%, and at zero where the gap is 0.  The
+   round count is an OWNERSHIP problem, not a colouring problem.
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
@@ -23,6 +69,11 @@ exactly one rank; owner range valid) before any metric is recorded.
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
 
@@ -128,6 +179,65 @@ def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
                      f"expected one of {METHODS}")
 
 
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
+    it (see the module docstring).  ``coloring_headroom_rounds`` is the
+    MAXIMUM number of rounds a perfect recolouring could still remove:
+    ``max(0, gap - 1)`` guaranteed, because Vizing allows the true optimum
+    to be ``max_degree + 1``.  It is deliberately NOT a "recolour vs
+    ownership" verdict — at ``gap == 1`` the colouring may already be
+    optimal (Class 2 graph) and deciding that is NP-complete.
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
+        # Rounds a perfect recolouring could still remove, GUARANTEED.
+        # Vizing permits the true optimum to be max_degree + 1, so a gap of
+        # 1 buys nothing provable; only gap >= 2 does.
+        "coloring_headroom_rounds": max(0, gap - 1),
+        "coloring_optimal_proven": gap == 0,
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
@@ -138,6 +248,18 @@ def main() -> int:
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
     p.add_argument("--out", type=str,
                    default="results/a1/voronoi_partition_quality.json")
     args = p.parse_args()
@@ -154,7 +276,8 @@ def main() -> int:
     from legoesm.grids.voronoi import create_voronoi_mesh
     from legoesm.parallel.voronoi_partition import resolve_partition_method
 
-    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
+    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
+                               lloyd_iterations=args.lloyd)
     if max(rank_counts) > int(mesh.nCells):
         raise SystemExit(
             f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
@@ -185,6 +308,22 @@ def main() -> int:
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
+                           f"<= {sc['coloring_headroom_rounds']} round(s) "
+                           f"of provable recolour headroom")
+                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
+                      f"rounds={sc['n_rounds']:3d} "
+                      f"max_degree={sc['max_degree']:3d} "
+                      f"gap={sc['coloring_gap']:+d} "
+                      f"-> {verdict} "
+                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
 
     payload = {
         "rows": rows,
@@ -204,7 +343,9 @@ def main() -> int:
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
diff --git a/tests/bench/test_bench_voronoi_partition_methods.py b/tests/bench/test_bench_voronoi_partition_methods.py
index 21f861fd7..538cc128c 100644
--- a/tests/bench/test_bench_voronoi_partition_methods.py
+++ b/tests/bench/test_bench_voronoi_partition_methods.py
@@ -146,3 +146,133 @@ def test_halo_matches_runtime_partition():
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
+    "n_rounds, max_degree, gap, headroom, proven",
+    [
+        # Provably optimal: no proper edge colouring beats max_degree.
+        (12, 12, 0, 0, True),
+        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
+        # buys nothing provable — this is the case that a naive
+        # "gap > 0 means recolour" rule would over-claim.
+        (13, 12, 1, 0, False),
+        # Only here is recolouring guaranteed to remove rounds, and at most
+        # gap-1 of them.
+        (14, 10, 4, 3, False),
+    ])
+def test_coloring_gap_and_headroom_are_derived_not_assumed(
+        monkeypatch, n_rounds, max_degree, gap, headroom, proven):
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
+    }
+    import legoesm.parallel.sharded_dynamics as sd
+    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
+
+    sc = mod.schedule_cost_row(object(), "geometric", 8)
+    assert sc["coloring_gap"] == gap
+    assert sc["coloring_headroom_rounds"] == headroom
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
+    # Mesh provenance: a lloyd=0 synthetic mesh must never be readable as a
+    # production SCVT receipt.
+    assert payload2["metadata"]["extra"]["lloyd_iterations"] == 50
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
   170	    "n_rounds, max_degree, gap, headroom, proven",
   171	    [
   172	        # Provably optimal: no proper edge colouring beats max_degree.
   173	        (12, 12, 0, 0, True),
   174	        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
   175	        # buys nothing provable — this is the case that a naive
   176	        # "gap > 0 means recolour" rule would over-claim.
   177	        (13, 12, 1, 0, False),
   178	        # Only here is recolouring guaranteed to remove rounds, and at most
   179	        # gap-1 of them.
   180	        (14, 10, 4, 3, False),
   181	    ])
   182	def test_coloring_gap_and_headroom_are_derived_not_assumed(
   183	        monkeypatch, n_rounds, max_degree, gap, headroom, proven):
   184	    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
   185	
   186	    Non-vacuity, the hard way: on every mesh small enough to test quickly the
   187	    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
   188	    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
   189	    time), so a real-mesh assertion cannot tell a correct subtraction from a
   190	    hardcoded ``0``; that exact mutation passed the first version of this
   191	    test.  Stubbing the production scorer with KNOWN values is what makes
   192	    the assertion able to fail.
   193	
   194	    The ``gap == 1`` row is the one that matters: the schedule is a proper
   195	    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
   196	    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
   197	    indistinguishable from optimal (Class 2), and claiming recolouring
   198	    headroom there would be an over-claim.
   199	    """
   200	    stub = {
   201	        "n_rounds": n_rounds, "max_degree": max_degree,
   202	        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
   203	        "resolved_method": "geometric", "halo_depth": 3,
   204	        "cells_per_device": 99_999, "production_strategy": "ppermute",
   205	    }
   206	    import legoesm.parallel.sharded_dynamics as sd
   207	    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
   208	
   209	    sc = mod.schedule_cost_row(object(), "geometric", 8)
   210	    assert sc["coloring_gap"] == gap
   211	    assert sc["coloring_headroom_rounds"] == headroom
   212	    assert sc["coloring_optimal_proven"] is proven
   213	
   214	
   215	def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
   216	    """Real-mesh sanity on the bound itself: a proper edge colouring can
   217	    never use fewer rounds than the graph's max degree, and the multi-start
   218	    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
   219	    fires, ``max_degree`` is not the degree of the graph being coloured and
   220	    every gap-based conclusion built on it is void."""
   221	    for method in ("geometric", "sfc"):
   222	        for n_ranks in (2, 4, 8):
   223	            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
   224	            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
   225	                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
   226	                f"max_degree={sc['max_degree']}")
   227	
   228	
   229	def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
   230	    """The schedule is scored at the SPMD production halo depth, NOT this
   231	    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
   232	    a different graph and quietly report the wrong lane's cost."""
   233	    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
   234	
   235	    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
   236	    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
   237	
   238	
   239	def test_schedule_cost_matches_the_production_scorer_exactly():
   240	    """Lock: the wrapper reports what the production scorer returns — it is
   241	    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
   242	    reports 8 rounds where the real depth-3 graph reports 12-14)."""
   243	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   244	
   245	    mesh = _mesh()
   246	    ref = spmd_schedule_cost(mesh, 2, method="sfc")
   247	    sc = mod.schedule_cost_row(mesh, "sfc", 2)
   248	    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
   249	                "coloring_method", "resolved_method", "cells_per_device",
   250	                "production_strategy"):
   251	        assert sc[key] == ref[key], key
   252	
   253	
   254	def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
   255	        tmp_path, monkeypatch):
   256	    """Off by default (it is the expensive layer); on, every scored row
   257	    carries the schedule block and the run records that it ran."""
   258	    out = tmp_path / "off.json"
   259	    monkeypatch.setattr(sys, "argv", [
   260	        "bench", "--subdivision", "2", "--rank-counts", "2",
   261	        "--methods", "geometric", "--out", str(out)])
   262	    assert mod.main() == 0
   263	    payload = json.loads(out.read_text())
   264	    assert "schedule" not in payload["rows"][0]
   265	    assert payload["metadata"]["extra"]["schedule_cost"] is False
   266	
   267	    out2 = tmp_path / "on.json"
   268	    monkeypatch.setattr(sys, "argv", [
   269	        "bench", "--subdivision", "2", "--rank-counts", "2",
   270	        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
   271	    assert mod.main() == 0
   272	    payload2 = json.loads(out2.read_text())
   273	    row = payload2["rows"][0]
   274	    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
   275	    assert payload2["metadata"]["extra"]["schedule_cost"] is True
   276	    # Mesh provenance: a lloyd=0 synthetic mesh must never be readable as a
   277	    # production SCVT receipt.
   278	    assert payload2["metadata"]["extra"]["lloyd_iterations"] == 50
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
    33	     headroom is exactly ZERO.  Only a partitioner with a new objective
    34	     (minimize boundary max-degree, not edge cut) can lower the round
    35	     count.
    36	   * ``gap == 1`` -> INDISTINGUISHABLE from optimal here.  A Class 2 graph
    37	     genuinely needs ``Delta + 1``, and deciding Class 1 vs Class 2 is
    38	     NP-complete, so this does NOT establish that a better colouring
    39	     exists.  Ceiling either way: one round.
    40	   * ``gap >= 2`` -> at least ``gap - 1`` rounds of real recolouring
    41	     headroom (Vizing caps the optimum at ``Delta + 1``).
    42	
    43	   Consequence worth stating before any measurement: wherever the
    44	   multi-start search lands on ``Delta`` or ``Delta + 1`` — which it does
    45	   on every configuration probed so far — recolouring is capped at ONE
    46	   round out of 12-14, i.e. <= ~7%, and at zero where the gap is 0.  The
    47	   round count is an OWNERSHIP problem, not a colouring problem.
    48	
    49	   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
    50	   flag is the MPI lane's (default 2), while the schedule is scored at the
    51	   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
    52	   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
    53	   evaluations of the integrator actually run.  When
    54	   ``production_strategy`` is ``"allgather"`` (auto-selected below the
    55	   cells/device threshold) there is no ppermute schedule in production and
    56	   the round count is COUNTERFACTUAL; the row says so.
    57	2. **Step time** (optional pointer, NOT run here): drive the existing
    58	   MPI lane with ``bench_ocean_mpas_scaling.py --partition-method <m>``
    59	   (ocean) or ``bench_mpas_spmd_scaling.py --partition-method <m>``
    60	   (atmosphere SPMD) — one method per launch, same case otherwise
    61	   (controlled comparison).
    62	
    63	Guards: methods that are unavailable (``metis`` without ``pymetis``) are
    64	reported as ``"unavailable"`` — never silently substituted, so a table
    65	column can never claim METIS numbers that actually came from the RCB
    66	fallback.  Partition CORRECTNESS is asserted per row (every cell owned by
    67	exactly one rank; owner range valid) before any metric is recorded.
    68	
    69	Run:
    70	  python scripts/bench/bench_voronoi_partition_methods.py \
    71	      --subdivision 6 --rank-counts 2,4,8,16 --out results/partition_quality.json
    72	
    73	  # + the SPMD schedule depth (minutes to hours at subdiv>=8 — batch it):
    74	  python scripts/bench/bench_voronoi_partition_methods.py \
    75	      --subdivision 9 --rank-counts 64,128 --schedule-cost \
    76	      --out results/a1/schedule_cost_s9.json
    77	"""
    78	from __future__ import annotations
    79	
    80	import argparse
    81	import json
    82	import os
    83	import sys
    84	from pathlib import Path
    85	
    86	import numpy as np
    87	
    88	sys.path.insert(0, str(Path(__file__).resolve().parent))
    89	
    90	from metadata import annotate_incomplete, scaling_metadata  # noqa: E402
    91	
    92	METHODS = ("geometric", "sfc", "metis")
    93	
    94	
    95	def method_available(method: str) -> bool:
    96	    if method != "metis":
    97	        return True
    98	    try:
    99	        import pymetis  # noqa: F401
   100	
   101	        return True
   102	    except Exception:
   103	        return False
   104	
   105	
   106	def partition_quality(mesh, cell_owner: np.ndarray, n_ranks: int,
   107	                      halo_depth: int = 2) -> dict:
   108	    """Exact partition-quality metrics from a global owner array.
   109	
   110	    Serial enumeration of every rank (no MPI): the same halo construction
   111	    the runtime uses (``compute_halo_cells``), so the reported halo sizes
   112	    are the runtime's, not an estimate.
   113	    """
   114	    from legoesm.parallel.voronoi_partition import compute_halo_cells
   115	
   116	    n_cells = int(mesh.nCells)
   117	    # Correctness: every cell owned exactly once, owners in range, no
   118	    # empty rank (an empty rank would silently deflate the halo/owned
   119	    # ratio through the max(counts, 1) guard).
   120	    if n_cells == 0 or cell_owner.size == 0:
   121	        raise AssertionError("empty mesh / owner array")
   122	    if cell_owner.shape != (n_cells,):
   123	        raise AssertionError(f"owner shape {cell_owner.shape} != ({n_cells},)")
   124	    if cell_owner.min() < 0 or cell_owner.max() >= n_ranks:
   125	        raise AssertionError("owner out of range")
   126	    counts = np.bincount(cell_owner, minlength=n_ranks).astype(float)
   127	    if int(counts.sum()) != n_cells:
   128	        raise AssertionError("ownership does not cover the mesh")
   129	    if counts.min() <= 0:
   130	        raise AssertionError(
   131	            f"empty rank in partition (counts.min()={counts.min():.0f}) — "
   132	            f"a skipped rank corrupts every per-rank metric")
   133	
   134	    # Edge cut: edges whose two cells have different owners.
   135	    c1, c2 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
   136	    valid = (c1 >= 0) & (c2 >= 0)
   137	    edge_cut = int((cell_owner[c1[valid]] != cell_owner[c2[valid]]).sum())
   138	
   139	    halo_sizes = []
   140	    neighbor_counts = []
   141	    cells_on_cell = np.asarray(mesh.cellsOnCell)
   142	    max_edges = int(cells_on_cell.shape[0]) if cells_on_cell.ndim == 2 else 0
   143	    for r in range(n_ranks):
   144	        halo = compute_halo_cells(
   145	            cell_owner, mesh.cellsOnCell, mesh.maxEdges, r, halo_depth)
   146	        halo_sizes.append(len(halo))
   147	        neighbor_counts.append(
   148	            len(set(int(cell_owner[c]) for c in halo) - {r}))
   149	    _ = max_edges
   150	    halo_sizes = np.array(halo_sizes, dtype=float)
   151	    return {
   152	        "cells_per_rank_min": int(counts.min()),
   153	        "cells_per_rank_max": int(counts.max()),
   154	        "load_imbalance_max_over_mean": float(counts.max() / counts.mean()),
   155	        "edge_cut": edge_cut,
   156	        "edge_cut_fraction": float(edge_cut / max(int(valid.sum()), 1)),
   157	        "halo_cells_max": int(halo_sizes.max()),
   158	        "halo_cells_mean": float(halo_sizes.mean()),
   159	        "halo_owned_ratio_max": float(
   160	            (halo_sizes / np.maximum(counts, 1.0)).max()),
   161	        "neighbor_ranks_max": int(max(neighbor_counts)),
   162	    }
   163	
   164	
   165	def owner_for(mesh, method: str, n_ranks: int) -> np.ndarray:
   166	    from legoesm.parallel.voronoi_partition import (
   167	        partition_cells_geometric,
   168	        partition_cells_metis,
   169	        partition_cells_sfc,
   170	    )
   171	
   172	    if method == "geometric":
   173	        return np.asarray(partition_cells_geometric(mesh, n_ranks))
   174	    if method == "sfc":
   175	        return np.asarray(partition_cells_sfc(mesh, n_ranks))
   176	    if method == "metis":
   177	        return np.asarray(partition_cells_metis(mesh, n_ranks))
   178	    raise ValueError(f"unknown partition method {method!r}; "
   179	                     f"expected one of {METHODS}")
   180	
   181	
   182	def schedule_cost_row(mesh, method: str, n_ranks: int) -> dict:
   183	    """SPMD halo-schedule depth for one (method, n_ranks) candidate.
   184	
   185	    Thin wrapper over the production
   186	    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
   187	    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
   188	    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
   189	    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
   190	    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
   191	    lane's (2), and scoring the SPMD schedule at 2 would colour a different
   192	    graph.
   193	
   194	    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
   195	    it (see the module docstring).  ``coloring_headroom_rounds`` is the
   196	    MAXIMUM number of rounds a perfect recolouring could still remove:
   197	    ``max(0, gap - 1)`` guaranteed, because Vizing allows the true optimum
   198	    to be ``max_degree + 1``.  It is deliberately NOT a "recolour vs
   199	    ownership" verdict — at ``gap == 1`` the colouring may already be
   200	    optimal (Class 2 graph) and deciding that is NP-complete.
   201	
   202	    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
   203	    different reorder target, which would mis-slice the owned blocks — is
   204	    unreachable from here: this passes the raw mesh with the scorer's default
   205	    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
   206	    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
   207	    the call would therefore only swallow *unforeseen* failures into a row
   208	    that reads like an orderly skip, which is how a missing number turns into
   209	    a silently wrong table.  Rows already print as the sweep goes, so a raise
   210	    keeps the completed rungs in the log.
   211	    """
   212	    import time
   213	
   214	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   215	
   216	    t0 = time.perf_counter()
   217	    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
   218	    gap = int(cost["n_rounds"]) - int(cost["max_degree"])
   219	    return {
   220	        "n_rounds": int(cost["n_rounds"]),
   221	        "n_rounds_greedy": int(cost["n_rounds_greedy"]),
   222	        "max_degree": int(cost["max_degree"]),
   223	        # The decisive column, read through Vizing (see module docstring).
   224	        "coloring_gap": gap,
   225	        # Rounds a perfect recolouring could still remove, GUARANTEED.
   226	        # Vizing permits the true optimum to be max_degree + 1, so a gap of
   227	        # 1 buys nothing provable; only gap >= 2 does.
   228	        "coloring_headroom_rounds": max(0, gap - 1),
   229	        "coloring_optimal_proven": gap == 0,
   230	        "coloring_method": cost["coloring_method"],
   231	        "resolved_method": cost["resolved_method"],
   232	        "schedule_halo_depth": int(cost["halo_depth"]),
   233	        "cells_per_device": int(cost["cells_per_device"]),
   234	        # "allgather" => production runs no ppermute schedule here, so the
   235	        # round count above is COUNTERFACTUAL, not a cost production pays.
   236	        "production_strategy": cost["production_strategy"],
   237	        "score_seconds": round(time.perf_counter() - t0, 2),
   238	    }
   239	
   240	
   241	def main() -> int:
   242	    p = argparse.ArgumentParser(
   243	        description=__doc__,
   244	        formatter_class=argparse.RawDescriptionHelpFormatter)
   245	    p.add_argument("--subdivision", type=int, default=5,
   246	                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
   247	    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
   248	    p.add_argument("--halo-depth", type=int, default=2,
   249	                   help="Halo layers (runtime default 2, del4 support).")
   250	    p.add_argument("--methods", type=str, default=",".join(METHODS))
   251	    p.add_argument("--schedule-cost", action="store_true",
   252	                   help="Also score the SPMD ppermute halo-schedule depth "
   253	                        "(n_rounds vs max_degree) per method x rank count. "
   254	                        "Uses the production SPMD halo depth, NOT "
   255	                        "--halo-depth. Expensive: minutes per candidate at "
   256	                        "subdiv>=8 — run it under batch.")
   257	    p.add_argument("--lloyd", type=int, default=50,
   258	                   help="Lloyd relaxation iterations for the mesh. 50 = the "
   259	                        "production SCVT key; 0 = the LABELLED synthetic "
   260	                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
   261	                        "masquerade as a production receipt, and it must "
   262	                        "match the prewarmed cache key at subdiv>=9.")
   263	    p.add_argument("--out", type=str,
   264	                   default="results/a1/voronoi_partition_quality.json")
   265	    args = p.parse_args()
   266	
   267	    rank_counts = [int(x) for x in args.rank_counts.split(",") if x]
   268	    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
   269	    for m in methods:
   270	        if m not in METHODS:
   271	            raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
   272	    if not rank_counts or any(n < 2 for n in rank_counts):
   273	        raise SystemExit("--rank-counts needs integers >= 2")
   274	
   275	
   276	    from legoesm.grids.voronoi import create_voronoi_mesh
   277	    from legoesm.parallel.voronoi_partition import resolve_partition_method
   278	
   279	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
   280	                               lloyd_iterations=args.lloyd)
   281	    if max(rank_counts) > int(mesh.nCells):
   282	        raise SystemExit(
   283	            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
   284	            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
   285	    print(f"mesh L{args.subdivision}: {int(mesh.nCells)} cells, "
   286	          f"{int(mesh.nEdges)} edges; auto -> "
   287	          f"{resolve_partition_method('auto')!r}")
   288	
   289	    rows = []
   290	    for method in methods:
   291	        if not method_available(method):
   292	            print(f"  {method:9s}: UNAVAILABLE (pymetis not importable) — "
   293	                  f"column omitted, never substituted")
   294	            rows.append({"method": method, "available": False})
   295	            continue
   296	        for n_ranks in rank_counts:
   297	            q = partition_quality(
   298	                mesh, owner_for(mesh, method, n_ranks), n_ranks,
   299	                halo_depth=args.halo_depth)
   300	            row = {"method": method, "available": True,
   301	                   "n_ranks": n_ranks, **q}
   302	            rows.append(row)
   303	            print(f"  {method:9s} np={n_ranks:3d} | "
   304	                  f"imbalance={q['load_imbalance_max_over_mean']:.3f} | "
   305	                  f"edge_cut={q['edge_cut']:6d} "
   306	                  f"({100 * q['edge_cut_fraction']:.2f}%) | "
   307	                  f"halo max={q['halo_cells_max']:5d} "
   308	                  f"mean={q['halo_cells_mean']:8.1f} | "
   309	                  f"halo/owned max={q['halo_owned_ratio_max']:.3f} | "
   310	                  f"nbrs max={q['neighbor_ranks_max']}")
   311	            if args.schedule_cost:
   312	                sc = schedule_cost_row(mesh, method, n_ranks)
   313	                row["schedule"] = sc
   314	                note = ("  [COUNTERFACTUAL: production auto-selects "
   315	                        "allgather here, no ppermute schedule]"
   316	                        if sc["production_strategy"] == "allgather" else "")
   317	                verdict = ("colouring PROVABLY OPTIMAL (Vizing floor)"
   318	                           if sc["coloring_optimal_proven"] else
   319	                           f"<= {sc['coloring_headroom_rounds']} round(s) "
   320	                           f"of provable recolour headroom")
   321	                print(f"    schedule(depth={sc['schedule_halo_depth']}): "
   322	                      f"rounds={sc['n_rounds']:3d} "
   323	                      f"max_degree={sc['max_degree']:3d} "
   324	                      f"gap={sc['coloring_gap']:+d} "
   325	                      f"-> {verdict} "
   326	                      f"({sc['score_seconds']:.1f}s){note}", flush=True)
   327	
   328	    payload = {
   329	        "rows": rows,
   330	        "auto_resolves_to": resolve_partition_method("auto"),
   331	        "step_time_pointer": (
   332	            "step-time per method: bench_ocean_mpas_scaling.py / "
   333	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   334	            "per launch, same case otherwise)"),
   335	        "metadata": annotate_incomplete(scaling_metadata(
   336	            grid="voronoi",
   337	            component="partitioning",
   338	            resolution=f"L{args.subdivision}",
   339	            n_levels=0,
   340	            precision="n/a",
   341	            decomposition="cell_partition",
   342	            solver_variant="n/a",
   343	            scaling_kind="partition-quality",
   344	            transport="none",
   345	            extra={"rank_counts": rank_counts, "methods": methods,
   346	                   "halo_depth": args.halo_depth,
   347	                   "schedule_cost": bool(args.schedule_cost),
   348	                   "lloyd_iterations": args.lloyd},
   349	        )),
   350	    }
   351	    outdir = os.path.dirname(args.out)
   352	    if outdir:
   353	        os.makedirs(outdir, exist_ok=True)
   354	    with open(args.out, "w") as f:
   355	        json.dump(payload, f, indent=2)
   356	    print(f"JSON: {args.out}")
   357	    return 0
   358	
   359	
   360	if __name__ == "__main__":
   361	    raise SystemExit(main())
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

codex
The current diff has already been revised to account for Vizing; I’m checking the remaining production-parity and provenance claims against the actual SPMD construction now.
exec
/usr/bin/bash -lc "rg -n -C 4 \"reorder_voronoi_for_sharding|reorder_target|create_voronoi_device_mesh|make_voronoi_sharded_step|SPMD_HALO_DEPTH|--halo-depth|lloyd\" scripts packages tests -g '*.py' -g '*.sbatch' -g '*.slurm' | head -n 800; rg -n -C 5 \"def create_voronoi_device_mesh|voronoi_dims|nCells.*nEdges|reorder\" packages/core/legoesm/parallel/mesh.py packages/core/legoesm -g '*.py' | head -n 600; rg -n -C 5 \"def scaling_metadata|def annotate_incomplete|scaling_metadata\\(\" scripts packages/core/legoesm -g '*.py' | head -n 400" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 342ms:
scripts/bench/bench_mpas_spmd_scaling.py-1-"""Strong scaling bench for the device-sharded icosahedral/MPAS (TRiSK)
scripts/bench/bench_mpas_spmd_scaling.py:2:hydrostatic atm step (``make_voronoi_sharded_step`` — cell-partition reorder +
scripts/bench/bench_mpas_spmd_scaling.py-3-ppermute halo).
scripts/bench/bench_mpas_spmd_scaling.py-4-
scripts/bench/bench_mpas_spmd_scaling.py-5-The Voronoi twin of ``bench_atm_latlon_spmd_scaling.py`` (mirrored
scripts/bench/bench_mpas_spmd_scaling.py-6-flag-for-flag where the grids allow): the global mesh is REORDERED with
scripts/bench/bench_mpas_spmd_scaling.py:7:``reorder_voronoi_for_sharding`` (METIS/RCB/Hilbert-SFC cell partition, ghost-
scripts/bench/bench_mpas_spmd_scaling.py-8-padded to an even device split) so each device's contiguous ``P("device")``
scripts/bench/bench_mpas_spmd_scaling.py-9-shard is a spatially compact cell cluster, then the SSP-RK3 step exchanges
scripts/bench/bench_mpas_spmd_scaling.py-10-only the partition-boundary halo per stage via ``jax.lax.ppermute``.
scripts/bench/bench_mpas_spmd_scaling.py-11-
--
scripts/bench/bench_mpas_spmd_scaling.py-30-
scripts/bench/bench_mpas_spmd_scaling.py-31-Multi-controller (route-B, ``--multicontroller``): identical contract to the
scripts/bench/bench_mpas_spmd_scaling.py-32-lat-lon benches — every process calls ``jax.distributed.initialize`` BEFORE
scripts/bench/bench_mpas_spmd_scaling.py-33-any other JAX use, the ("device",) mesh is built over the GLOBAL
scripts/bench/bench_mpas_spmd_scaling.py:34:``jax.devices()``, and the existing ``make_voronoi_sharded_step`` ppermute
scripts/bench/bench_mpas_spmd_scaling.py-35-halo + the mass-fix psum run unchanged across processes (NCCL on GPU / gloo
scripts/bench/bench_mpas_spmd_scaling.py-36-on CPU). NO mpi4jax is armed in this mode (the documented mixed-stack
scripts/bench/bench_mpas_spmd_scaling.py-37-deadlock hazard). Every process computes the SAME reorder host-side; under
scripts/bench/bench_mpas_spmd_scaling.py-38-``--multicontroller`` the partition checksum is asserted equal across
--
scripts/bench/bench_mpas_spmd_scaling.py-95-# is the allreduce rounding floor, not scheme drift.
scripts/bench/bench_mpas_spmd_scaling.py-96-MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
scripts/bench/bench_mpas_spmd_scaling.py-97-
scripts/bench/bench_mpas_spmd_scaling.py-98-
scripts/bench/bench_mpas_spmd_scaling.py:99:def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
scripts/bench/bench_mpas_spmd_scaling.py:100:                          moist=False, lloyd_iterations=50):
scripts/bench/bench_mpas_spmd_scaling.py-101-    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
scripts/bench/bench_mpas_spmd_scaling.py-102-
scripts/bench/bench_mpas_spmd_scaling.py:103:    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
scripts/bench/bench_mpas_spmd_scaling.py-104-    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
scripts/bench/bench_mpas_spmd_scaling.py-105-    device count of THIS run's mesh/model (the two differ for the
scripts/bench/bench_mpas_spmd_scaling.py-106-    single-device reference leg of a ladder, via ``--reorder-for``).
scripts/bench/bench_mpas_spmd_scaling.py-107-    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
--
scripts/bench/bench_mpas_spmd_scaling.py-113-        MPASPrimitiveEquationModel,
scripts/bench/bench_mpas_spmd_scaling.py-114-    )
scripts/bench/bench_mpas_spmd_scaling.py-115-    from legoesm.grids.vertical import create_sigma_coordinate
scripts/bench/bench_mpas_spmd_scaling.py-116-    from legoesm.grids.voronoi import create_voronoi_mesh
scripts/bench/bench_mpas_spmd_scaling.py:117:    from legoesm.parallel.mesh import create_voronoi_device_mesh
scripts/bench/bench_mpas_spmd_scaling.py:118:    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
scripts/bench/bench_mpas_spmd_scaling.py-119-
scripts/bench/bench_mpas_spmd_scaling.py-120-    mesh = create_voronoi_mesh(subdivision_level=subdivision,
scripts/bench/bench_mpas_spmd_scaling.py:121:                               lloyd_iterations=lloyd_iterations)
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
scripts/bench/bench_mpas_spmd_scaling.py:137:    dev_config = create_voronoi_device_mesh(
scripts/bench/bench_mpas_spmd_scaling.py-138-        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
scripts/bench/bench_mpas_spmd_scaling.py-139-        n_devices=run_nd,
scripts/bench/bench_mpas_spmd_scaling.py-140-    )
scripts/bench/bench_mpas_spmd_scaling.py-141-    if dev_config.n_devices > 1:
--
scripts/bench/bench_mpas_spmd_scaling.py-179-    p.add_argument("--subdivision", type=int, default=5,
scripts/bench/bench_mpas_spmd_scaling.py-180-                   help="icosahedral subdivision level L "
scripts/bench/bench_mpas_spmd_scaling.py-181-                        "(nCells = 10*4^L + 2 before ghost padding)")
scripts/bench/bench_mpas_spmd_scaling.py-182-    p.add_argument("--nlev", type=int, default=8)
scripts/bench/bench_mpas_spmd_scaling.py:183:    p.add_argument("--lloyd", type=int, default=50,
scripts/bench/bench_mpas_spmd_scaling.py-184-                   help="Lloyd relaxation iterations for the mesh. 50 = "
scripts/bench/bench_mpas_spmd_scaling.py-185-                        "production SCVT; 0 = labelled synthetic scaling "
scripts/bench/bench_mpas_spmd_scaling.py-186-                        "mesh (scaling receipts only, never physics — "
scripts/bench/bench_mpas_spmd_scaling.py-187-                        "must match the prewarmed cache key at subdiv>=9).")
--
scripts/bench/bench_mpas_spmd_scaling.py-204-                        "tracer fields.")
scripts/bench/bench_mpas_spmd_scaling.py-205-    p.add_argument("--halo-strategy",
scripts/bench/bench_mpas_spmd_scaling.py-206-                   choices=["auto", "ppermute", "allgather"],
scripts/bench/bench_mpas_spmd_scaling.py-207-                   default="auto",
scripts/bench/bench_mpas_spmd_scaling.py:208:                   help="Halo strategy for make_voronoi_sharded_step. "
scripts/bench/bench_mpas_spmd_scaling.py-209-                        "'auto' picks allgather below the per-device "
scripts/bench/bench_mpas_spmd_scaling.py-210-                        "cell threshold — force 'ppermute' to exercise "
scripts/bench/bench_mpas_spmd_scaling.py-211-                        "the neighbor-round schedule on small gate "
scripts/bench/bench_mpas_spmd_scaling.py-212-                        "meshes (the multicontroller selfspawn tests "
--
scripts/bench/bench_mpas_spmd_scaling.py-284-            init_jax_distributed_with_fallback()
scripts/bench/bench_mpas_spmd_scaling.py-285-
scripts/bench/bench_mpas_spmd_scaling.py-286-    from legoesm.parallel.sharded_dynamics import (
scripts/bench/bench_mpas_spmd_scaling.py-287-        gather_voronoi_state_spmd,
scripts/bench/bench_mpas_spmd_scaling.py:288:        make_voronoi_sharded_step,
scripts/bench/bench_mpas_spmd_scaling.py-289-    )
scripts/bench/bench_mpas_spmd_scaling.py-290-
scripts/bench/bench_mpas_spmd_scaling.py-291-    nd = args.n_devices
scripts/bench/bench_mpas_spmd_scaling.py-292-    avail = len(jax.devices())
--
scripts/bench/bench_mpas_spmd_scaling.py-312-            f"the ghost padding only guarantees divisibility for the "
scripts/bench/bench_mpas_spmd_scaling.py-313-            f"partition target.")
scripts/bench/bench_mpas_spmd_scaling.py-314-    mesh, model, s0, dev_config = build_model_and_state(
scripts/bench/bench_mpas_spmd_scaling.py-315-        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
scripts/bench/bench_mpas_spmd_scaling.py:316:        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
scripts/bench/bench_mpas_spmd_scaling.py-317-
scripts/bench/bench_mpas_spmd_scaling.py-318-    if args.multicontroller:
scripts/bench/bench_mpas_spmd_scaling.py-319-        # Every process computed the reorder independently — assert the
scripts/bench/bench_mpas_spmd_scaling.py-320-        # partitions agree before any collective uses the halo schedule.
--
scripts/bench/bench_mpas_spmd_scaling.py-387-    mass_before = None
scripts/bench/bench_mpas_spmd_scaling.py-388-    if args.check_conservation:
scripts/bench/bench_mpas_spmd_scaling.py-389-        mass_before = _global_dry_mass(s0_global, mesh)
scripts/bench/bench_mpas_spmd_scaling.py-390-
scripts/bench/bench_mpas_spmd_scaling.py:391:    step = make_voronoi_sharded_step(
scripts/bench/bench_mpas_spmd_scaling.py-392-        model, dev_config, halo_strategy=args.halo_strategy)
scripts/bench/bench_mpas_spmd_scaling.py-393-    # Already in the sharded layout (partition-local build) for nd > 1;
scripts/bench/bench_mpas_spmd_scaling.py-394-    # single-device s0 is the plain global state.
scripts/bench/bench_mpas_spmd_scaling.py-395-    s = s0
--
scripts/bench/bench_mpas_spmd_scaling.py-497-        component="mpas_atm",
scripts/bench/bench_mpas_spmd_scaling.py-498-        subdivision=args.subdivision, n_devices=nd,
scripts/bench/bench_mpas_spmd_scaling.py-499-        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py-500-        partition_method=args.partition_method, physics=args.physics,
scripts/bench/bench_mpas_spmd_scaling.py:501:        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
scripts/bench/bench_mpas_spmd_scaling.py-502-        # a row without this field could pass as a production-SCVT receipt.
scripts/bench/bench_mpas_spmd_scaling.py:503:        lloyd_iterations=args.lloyd,
scripts/bench/bench_mpas_spmd_scaling.py-504-        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
scripts/bench/bench_mpas_spmd_scaling.py-505-        # saying "auto" would not reveal whether ppermute or allgather
scripts/bench/bench_mpas_spmd_scaling.py-506-        # was actually measured (codex M3c-2 MINOR).
scripts/bench/bench_mpas_spmd_scaling.py-507-        halo_strategy_requested=args.halo_strategy,
--
tests/test_cases/baroclinic_wave.py-798-    build" applies to the O(nCells*nlev) STATE leaves; O(nEntities) 1-D
tests/test_cases/baroclinic_wave.py-799-    temporaries (``cos(angleEdge)``) are still evaluated in full, like the
tests/test_cases/baroclinic_wave.py-800-    global mesh itself.
tests/test_cases/baroclinic_wave.py-801-
tests/test_cases/baroclinic_wave.py:802:    Requires ``reorder_voronoi_for_sharding`` padding (nCells and nEdges
tests/test_cases/baroclinic_wave.py-803-    divisible by the device count) — the same precondition
tests/test_cases/baroclinic_wave.py-804-    ``shard_pytree`` has.  ``dev_config.face_sharding is None`` (single
tests/test_cases/baroclinic_wave.py-805-    device) falls back to the global builder unchanged.
tests/test_cases/baroclinic_wave.py-806-    """
--
tests/bench/test_bench_mpas_spmd_gates.py-55-    # Parser accepts the gate flags (argparse would SystemExit on unknowns).
tests/bench/test_bench_mpas_spmd_gates.py-56-    src = Path(_BENCH).read_text()
tests/bench/test_bench_mpas_spmd_gates.py-57-    for flag in ("--parity-gate", "--check-conservation", "--mass-rtol",
tests/bench/test_bench_mpas_spmd_gates.py-58-                 "--multicontroller", "--coordinator", "--partition-method",
tests/bench/test_bench_mpas_spmd_gates.py:59:                 "--reorder-for", "--lloyd"):
tests/bench/test_bench_mpas_spmd_gates.py-60-        assert flag in src
tests/bench/test_bench_mpas_spmd_gates.py-61-
tests/bench/test_bench_mpas_spmd_gates.py-62-
tests/bench/test_bench_mpas_spmd_gates.py:63:def test_lloyd_flag_reaches_the_mesh_builder(monkeypatch):
tests/bench/test_bench_mpas_spmd_gates.py:64:    """``--lloyd 0`` must select the synthetic scaling mesh, not lloyd=50.
tests/bench/test_bench_mpas_spmd_gates.py-65-
tests/bench/test_bench_mpas_spmd_gates.py-66-    Non-vacuous by construction: the sentinel records the kwarg
tests/bench/test_bench_mpas_spmd_gates.py-67-    ``create_voronoi_mesh`` actually receives, so dropping the plumbing
tests/bench/test_bench_mpas_spmd_gates.py-68-    (the state before this flag existed) makes the assertion fail rather
--
tests/bench/test_bench_mpas_spmd_gates.py-79-        raise RuntimeError("stop-after-mesh-request")
tests/bench/test_bench_mpas_spmd_gates.py-80-
tests/bench/test_bench_mpas_spmd_gates.py-81-    monkeypatch.setattr(voronoi, "create_voronoi_mesh", _spy)
tests/bench/test_bench_mpas_spmd_gates.py-82-    with pytest.raises(RuntimeError, match="stop-after-mesh-request"):
tests/bench/test_bench_mpas_spmd_gates.py:83:        mod.build_model_and_state(4, 4, 1, 1, "sfc", lloyd_iterations=0)
tests/bench/test_bench_mpas_spmd_gates.py-84-    assert seen["level"] == 4
tests/bench/test_bench_mpas_spmd_gates.py:85:    assert seen["lloyd_iterations"] == 0
tests/bench/test_bench_mpas_spmd_gates.py-86-
tests/bench/test_bench_mpas_spmd_gates.py-87-
tests/bench/test_bench_mpas_spmd_gates.py-88-def test_gather_voronoi_state_spmd_round_trip():
tests/bench/test_bench_mpas_spmd_gates.py-89-    """Direct exercise of the new gather: shard -> gather == original."""
--
tests/bench/test_bench_mpas_spmd_gates.py-93-    import jax.numpy as jnp
tests/bench/test_bench_mpas_spmd_gates.py-94-    import numpy as np
tests/bench/test_bench_mpas_spmd_gates.py-95-    from legoesm.grids.vertical import create_sigma_coordinate
tests/bench/test_bench_mpas_spmd_gates.py-96-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/bench/test_bench_mpas_spmd_gates.py:97:    from legoesm.parallel.mesh import create_voronoi_device_mesh, shard_pytree
tests/bench/test_bench_mpas_spmd_gates.py-98-    from legoesm.parallel.sharded_dynamics import gather_voronoi_state_spmd
tests/bench/test_bench_mpas_spmd_gates.py-99-    from legoesm.parallel.voronoi_partition import (
tests/bench/test_bench_mpas_spmd_gates.py:100:        reorder_voronoi_for_sharding,
tests/bench/test_bench_mpas_spmd_gates.py-101-    )
tests/bench/test_bench_mpas_spmd_gates.py-102-
tests/bench/test_bench_mpas_spmd_gates.py-103-    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
tests/bench/test_bench_mpas_spmd_gates.py-104-    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
tests/bench/test_bench_mpas_spmd_gates.py-105-
tests/bench/test_bench_mpas_spmd_gates.py:106:    mesh = reorder_voronoi_for_sharding(
tests/bench/test_bench_mpas_spmd_gates.py-107-        create_voronoi_mesh(subdivision_level=3), 2, method="sfc")
tests/bench/test_bench_mpas_spmd_gates.py-108-    state = baroclinic_wave_init_mpas(
tests/bench/test_bench_mpas_spmd_gates.py-109-        mesh, create_sigma_coordinate(4), perturbed=True)
tests/bench/test_bench_mpas_spmd_gates.py:110:    dev = create_voronoi_device_mesh(
tests/bench/test_bench_mpas_spmd_gates.py-111-        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
tests/bench/test_bench_mpas_spmd_gates.py-112-        n_devices=2)
tests/bench/test_bench_mpas_spmd_gates.py-113-    sharded = shard_pytree(state, dev)
tests/bench/test_bench_mpas_spmd_gates.py-114-    back = gather_voronoi_state_spmd(sharded, dev)
--
tests/bench/test_bench_mpas_spmd_gates.py-117-            np.asarray(getattr(back, name).data),
tests/bench/test_bench_mpas_spmd_gates.py-118-            np.asarray(getattr(state, name).data),
tests/bench/test_bench_mpas_spmd_gates.py-119-            err_msg=f"{name}: shard->gather round-trip not identity")
tests/bench/test_bench_mpas_spmd_gates.py-120-    # Single-device config passes through unchanged.
tests/bench/test_bench_mpas_spmd_gates.py:121:    dev1 = create_voronoi_device_mesh(
tests/bench/test_bench_mpas_spmd_gates.py-122-        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
tests/bench/test_bench_mpas_spmd_gates.py-123-        n_devices=1)
tests/bench/test_bench_mpas_spmd_gates.py-124-    assert gather_voronoi_state_spmd(state, dev1) is state
tests/bench/test_bench_mpas_spmd_gates.py-125-    assert isinstance(jnp.asarray(0.0), object)  # keep jnp import honest
--
tests/bench/test_bench_voronoi_partition_methods.py-184-    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
tests/bench/test_bench_voronoi_partition_methods.py-185-
tests/bench/test_bench_voronoi_partition_methods.py-186-    Non-vacuity, the hard way: on every mesh small enough to test quickly the
tests/bench/test_bench_voronoi_partition_methods.py-187-    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
tests/bench/test_bench_voronoi_partition_methods.py:188:    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
tests/bench/test_bench_voronoi_partition_methods.py-189-    time), so a real-mesh assertion cannot tell a correct subtraction from a
tests/bench/test_bench_voronoi_partition_methods.py-190-    hardcoded ``0``; that exact mutation passed the first version of this
tests/bench/test_bench_voronoi_partition_methods.py-191-    test.  Stubbing the production scorer with KNOWN values is what makes
tests/bench/test_bench_voronoi_partition_methods.py-192-    the assertion able to fail.
--
tests/bench/test_bench_voronoi_partition_methods.py-227-
tests/bench/test_bench_voronoi_partition_methods.py-228-
tests/bench/test_bench_voronoi_partition_methods.py-229-def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
tests/bench/test_bench_voronoi_partition_methods.py-230-    """The schedule is scored at the SPMD production halo depth, NOT this
tests/bench/test_bench_voronoi_partition_methods.py:231:    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
tests/bench/test_bench_voronoi_partition_methods.py-232-    a different graph and quietly report the wrong lane's cost."""
tests/bench/test_bench_voronoi_partition_methods.py:233:    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
tests/bench/test_bench_voronoi_partition_methods.py-234-
tests/bench/test_bench_voronoi_partition_methods.py-235-    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
tests/bench/test_bench_voronoi_partition_methods.py:236:    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
tests/bench/test_bench_voronoi_partition_methods.py-237-
tests/bench/test_bench_voronoi_partition_methods.py-238-
tests/bench/test_bench_voronoi_partition_methods.py-239-def test_schedule_cost_matches_the_production_scorer_exactly():
tests/bench/test_bench_voronoi_partition_methods.py-240-    """Lock: the wrapper reports what the production scorer returns — it is
--
tests/bench/test_bench_voronoi_partition_methods.py-272-    payload2 = json.loads(out2.read_text())
tests/bench/test_bench_voronoi_partition_methods.py-273-    row = payload2["rows"][0]
tests/bench/test_bench_voronoi_partition_methods.py-274-    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
tests/bench/test_bench_voronoi_partition_methods.py-275-    assert payload2["metadata"]["extra"]["schedule_cost"] is True
tests/bench/test_bench_voronoi_partition_methods.py:276:    # Mesh provenance: a lloyd=0 synthetic mesh must never be readable as a
tests/bench/test_bench_voronoi_partition_methods.py-277-    # production SCVT receipt.
tests/bench/test_bench_voronoi_partition_methods.py:278:    assert payload2["metadata"]["extra"]["lloyd_iterations"] == 50
--
tests/validation/test_cross_grid_rce_smoke.py-321-        compute_terrain_metric, create_height_coordinate,
tests/validation/test_cross_grid_rce_smoke.py-322-    )
tests/validation/test_cross_grid_rce_smoke.py-323-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/validation/test_cross_grid_rce_smoke.py-324-
tests/validation/test_cross_grid_rce_smoke.py:325:    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
tests/validation/test_cross_grid_rce_smoke.py-326-    hc = create_height_coordinate(NLEV, H_TOP)
tests/validation/test_cross_grid_rce_smoke.py-327-    tm = compute_terrain_metric(jnp.zeros(mesh.nCells), hc)
tests/validation/test_cross_grid_rce_smoke.py-328-    cfg = MPASCompressibleEulerConfig(
tests/validation/test_cross_grid_rce_smoke.py-329-        nu_del2=1.0e4, n_acoustic_substeps=4, fix_mass=False,
--
tests/validation/test_cross_grid_nh_consistency.py-244-        create_height_coordinate,
tests/validation/test_cross_grid_nh_consistency.py-245-    )
tests/validation/test_cross_grid_nh_consistency.py-246-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/validation/test_cross_grid_nh_consistency.py-247-
tests/validation/test_cross_grid_nh_consistency.py:248:    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
tests/validation/test_cross_grid_nh_consistency.py-249-    hc = create_height_coordinate(NLEV, H_TOP)
tests/validation/test_cross_grid_nh_consistency.py-250-    tm = compute_terrain_metric(
tests/validation/test_cross_grid_nh_consistency.py-251-        jnp.zeros(mesh.nCells, jnp.float64), hc,
tests/validation/test_cross_grid_nh_consistency.py-252-    )
--
tests/ocean/run_ocean_all_grids_matrix.py-247-
tests/ocean/run_ocean_all_grids_matrix.py-248-    parser.add_argument("--mpas-base-mesh-level", type=int, default=2)
tests/ocean/run_ocean_all_grids_matrix.py-249-    parser.add_argument("--mpas-dt", type=float, default=60.0)
tests/ocean/run_ocean_all_grids_matrix.py-250-    parser.add_argument("--save-every", type=int, default=10)
tests/ocean/run_ocean_all_grids_matrix.py:251:    parser.add_argument("--lloyd-iterations", type=int, default=30)
tests/ocean/run_ocean_all_grids_matrix.py-252-    parser.add_argument("--latlon-nlon", type=int, default=360)
tests/ocean/run_ocean_all_grids_matrix.py-253-    parser.add_argument("--latlon-nlat", type=int, default=181)
tests/ocean/run_ocean_all_grids_matrix.py-254-
tests/ocean/run_ocean_all_grids_matrix.py-255-    parser.add_argument("--days", type=float, default=5.0)
--
tests/distributed/test_voronoi_mpi.py-80-
tests/distributed/test_voronoi_mpi.py-81-
tests/distributed/test_voronoi_mpi.py-82-@pytest.fixture(scope="module")
tests/distributed/test_voronoi_mpi.py-83-def mesh():
tests/distributed/test_voronoi_mpi.py:84:    return create_voronoi_mesh(subdivision_level=SUBDIVISION_LEVEL, lloyd_iterations=5)
tests/distributed/test_voronoi_mpi.py-85-
tests/distributed/test_voronoi_mpi.py-86-
tests/distributed/test_voronoi_mpi.py-87-@pytest.fixture
tests/distributed/test_voronoi_mpi.py-88-def sigma():
--
tests/distributed/test_mpi_differentiability.py-342-            partition_cells_geometric,
tests/distributed/test_mpi_differentiability.py-343-            partition_voronoi_mesh,
tests/distributed/test_mpi_differentiability.py-344-        )
tests/distributed/test_mpi_differentiability.py-345-
tests/distributed/test_mpi_differentiability.py:346:        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/distributed/test_mpi_differentiability.py-347-        n_ranks = MPI.COMM_WORLD.Get_size()
tests/distributed/test_mpi_differentiability.py-348-        cell_owner = partition_cells_geometric(mesh, n_ranks)
tests/distributed/test_mpi_differentiability.py-349-        parts = [
tests/distributed/test_mpi_differentiability.py-350-            partition_voronoi_mesh(mesh, n_ranks, r, cell_owner=cell_owner)
--
scripts/bench/run_levante_gpu_scaling.py-492-    """Return valid GPU counts up to max_gpus for the given grid type.
scripts/bench/run_levante_gpu_scaling.py-493-
scripts/bench/run_levante_gpu_scaling.py-494-    Cubed-sphere requires divisors of 6 (face sharding) or 6*k^2 (tiling).
scripts/bench/run_levante_gpu_scaling.py-495-    Icosahedral supports any GPU count — its single-process multi-GPU path
scripts/bench/run_levante_gpu_scaling.py:496:    routes through ``make_voronoi_sharded_step``, a real domain-decomposed
scripts/bench/run_levante_gpu_scaling.py-497-    ``shard_map`` step.  Lat-lon and spectral are single-GPU only HERE
scripts/bench/run_levante_gpu_scaling.py-498-    (single-process): neither has a validated single-process multi-GPU
scripts/bench/run_levante_gpu_scaling.py-499-    sharded step.  Lat-lon's *multi-rank* scaling instead goes through MPI
scripts/bench/run_levante_gpu_scaling.py-500-    (``make_latlon_mpi_step``), where the count is pinned to world_size by
--
scripts/bench/run_levante_gpu_scaling.py-1457-
scripts/bench/run_levante_gpu_scaling.py-1458-    from legoesm.grids.vertical import create_sigma_coordinate
scripts/bench/run_levante_gpu_scaling.py-1459-    from legoesm.parallel.mesh import (
scripts/bench/run_levante_gpu_scaling.py-1460-        create_device_mesh, create_latlon_mesh, create_level_mesh,
scripts/bench/run_levante_gpu_scaling.py:1461:        create_voronoi_device_mesh, shard_pytree,
scripts/bench/run_levante_gpu_scaling.py-1462-    )
scripts/bench/run_levante_gpu_scaling.py-1463-
scripts/bench/run_levante_gpu_scaling.py-1464-    # Choose timestep
scripts/bench/run_levante_gpu_scaling.py-1465-    if dt is None:
--
scripts/bench/run_levante_gpu_scaling.py-1540-            model = MPASPrimitiveEquationModel(grid, sigma, config)
scripts/bench/run_levante_gpu_scaling.py-1541-            state = baroclinic_wave_init_mpas(grid, sigma, perturbed=True, moist=_moist)
scripts/bench/run_levante_gpu_scaling.py-1542-            # Each MPI rank uses 1 GPU (affinity set by
scripts/bench/run_levante_gpu_scaling.py-1543-            # _configure_mpi_gpu_affinity).
scripts/bench/run_levante_gpu_scaling.py:1544:            dev_config = create_voronoi_device_mesh(
scripts/bench/run_levante_gpu_scaling.py-1545-                nCells=_voronoi_layout.local_mesh.nCells,
scripts/bench/run_levante_gpu_scaling.py-1546-                nEdges=_voronoi_layout.local_mesh.nEdges,
scripts/bench/run_levante_gpu_scaling.py-1547-                nVertices=_voronoi_layout.local_mesh.nVertices,
scripts/bench/run_levante_gpu_scaling.py-1548-                n_devices=1,
--
scripts/bench/run_levante_gpu_scaling.py-1550-        else:
scripts/bench/run_levante_gpu_scaling.py-1551-            # Single-node: reorder for spatial locality and shard.
scripts/bench/run_levante_gpu_scaling.py-1552-            if n_gpus > 1:
scripts/bench/run_levante_gpu_scaling.py-1553-                from legoesm.parallel.voronoi_partition import (
scripts/bench/run_levante_gpu_scaling.py:1554:                    reorder_voronoi_for_sharding,
scripts/bench/run_levante_gpu_scaling.py-1555-                )
scripts/bench/run_levante_gpu_scaling.py:1556:                grid = reorder_voronoi_for_sharding(grid, n_gpus)
scripts/bench/run_levante_gpu_scaling.py-1557-
scripts/bench/run_levante_gpu_scaling.py:1558:            dev_config = create_voronoi_device_mesh(
scripts/bench/run_levante_gpu_scaling.py-1559-                nCells=grid.nCells,
scripts/bench/run_levante_gpu_scaling.py-1560-                nEdges=grid.nEdges,
scripts/bench/run_levante_gpu_scaling.py-1561-                nVertices=grid.nVertices,
scripts/bench/run_levante_gpu_scaling.py-1562-                n_devices=n_gpus,
--
scripts/bench/run_levante_gpu_scaling.py-1824-            model, _latlon_layout, physics_fn=physics_fn,
scripts/bench/run_levante_gpu_scaling.py-1825-        )
scripts/bench/run_levante_gpu_scaling.py-1826-    # SPMD sharded step functions (single-node multi-GPU).
scripts/bench/run_levante_gpu_scaling.py-1827-    elif grid_type == "icosahedral" and dev_config.n_devices > 1:
scripts/bench/run_levante_gpu_scaling.py:1828:        from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step
scripts/bench/run_levante_gpu_scaling.py:1829:        step_fn = make_voronoi_sharded_step(model, dev_config)
scripts/bench/run_levante_gpu_scaling.py-1830-    elif grid_type == "cubed-sphere" and dev_config.n_devices > 1:
scripts/bench/run_levante_gpu_scaling.py-1831-        from legoesm.parallel.sharded_dynamics import make_sharded_step
scripts/bench/run_levante_gpu_scaling.py-1832-        step_fn = make_sharded_step(model, dev_config, n=n_grid, nlev=n_levels)
scripts/bench/run_levante_gpu_scaling.py-1833-        # ``make_sharded_step`` itself activates the explicit SPMD halo
--
scripts/bench/bench_voronoi_partition_methods.py-45-   on every configuration probed so far — recolouring is capped at ONE
scripts/bench/bench_voronoi_partition_methods.py-46-   round out of 12-14, i.e. <= ~7%, and at zero where the gap is 0.  The
scripts/bench/bench_voronoi_partition_methods.py-47-   round count is an OWNERSHIP problem, not a colouring problem.
scripts/bench/bench_voronoi_partition_methods.py-48-
scripts/bench/bench_voronoi_partition_methods.py:49:   NOTE the halo depth differs by lane and is NOT ``--halo-depth``: that
scripts/bench/bench_voronoi_partition_methods.py-50-   flag is the MPI lane's (default 2), while the schedule is scored at the
scripts/bench/bench_voronoi_partition_methods.py:51:   SPMD production depth ``SPMD_HALO_DEPTH`` (3).  Both are recorded.
scripts/bench/bench_voronoi_partition_methods.py-52-   ``n_rounds`` is per HALO FILL, not per step — multiply by the tendency
scripts/bench/bench_voronoi_partition_methods.py-53-   evaluations of the integrator actually run.  When
scripts/bench/bench_voronoi_partition_methods.py-54-   ``production_strategy`` is ``"allgather"`` (auto-selected below the
scripts/bench/bench_voronoi_partition_methods.py-55-   cells/device threshold) there is no ppermute schedule in production and
--
scripts/bench/bench_voronoi_partition_methods.py-184-
scripts/bench/bench_voronoi_partition_methods.py-185-    Thin wrapper over the production
scripts/bench/bench_voronoi_partition_methods.py-186-    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
scripts/bench/bench_voronoi_partition_methods.py-187-    the RAW mesh for ``n_ranks`` with ``method`` and colours the real
scripts/bench/bench_voronoi_partition_methods.py:188:    depth-``SPMD_HALO_DEPTH`` communication graph, so the number is the one
scripts/bench/bench_voronoi_partition_methods.py-189-    production pays, not a 1-ring lookalike.  ``halo_depth`` is deliberately
scripts/bench/bench_voronoi_partition_methods.py:190:    LEFT AT THE SCORER'S DEFAULT: this script's ``--halo-depth`` is the MPI
scripts/bench/bench_voronoi_partition_methods.py-191-    lane's (2), and scoring the SPMD schedule at 2 would colour a different
scripts/bench/bench_voronoi_partition_methods.py-192-    graph.
scripts/bench/bench_voronoi_partition_methods.py-193-
scripts/bench/bench_voronoi_partition_methods.py-194-    Adds ``coloring_gap = n_rounds - max_degree`` and the Vizing reading of
--
scripts/bench/bench_voronoi_partition_methods.py-201-
scripts/bench/bench_voronoi_partition_methods.py-202-    Errors are NOT caught.  The scorer's one refusal — a mesh padded for a
scripts/bench/bench_voronoi_partition_methods.py-203-    different reorder target, which would mis-slice the owned blocks — is
scripts/bench/bench_voronoi_partition_methods.py-204-    unreachable from here: this passes the raw mesh with the scorer's default
scripts/bench/bench_voronoi_partition_methods.py:205:    ``reorder_target = n_ranks``, and ``reorder_voronoi_for_sharding`` pads
scripts/bench/bench_voronoi_partition_methods.py-206-    ``nCells``/``nEdges`` to be divisible by exactly that target.  Wrapping
scripts/bench/bench_voronoi_partition_methods.py-207-    the call would therefore only swallow *unforeseen* failures into a row
scripts/bench/bench_voronoi_partition_methods.py-208-    that reads like an orderly skip, which is how a missing number turns into
scripts/bench/bench_voronoi_partition_methods.py-209-    a silently wrong table.  Rows already print as the sweep goes, so a raise
--
scripts/bench/bench_voronoi_partition_methods.py-244-        formatter_class=argparse.RawDescriptionHelpFormatter)
scripts/bench/bench_voronoi_partition_methods.py-245-    p.add_argument("--subdivision", type=int, default=5,
scripts/bench/bench_voronoi_partition_methods.py-246-                   help="Icosahedral level (L5=10,242 cells; L6=40,962).")
scripts/bench/bench_voronoi_partition_methods.py-247-    p.add_argument("--rank-counts", type=str, default="2,4,8,16")
scripts/bench/bench_voronoi_partition_methods.py:248:    p.add_argument("--halo-depth", type=int, default=2,
scripts/bench/bench_voronoi_partition_methods.py-249-                   help="Halo layers (runtime default 2, del4 support).")
scripts/bench/bench_voronoi_partition_methods.py-250-    p.add_argument("--methods", type=str, default=",".join(METHODS))
scripts/bench/bench_voronoi_partition_methods.py-251-    p.add_argument("--schedule-cost", action="store_true",
scripts/bench/bench_voronoi_partition_methods.py-252-                   help="Also score the SPMD ppermute halo-schedule depth "
scripts/bench/bench_voronoi_partition_methods.py-253-                        "(n_rounds vs max_degree) per method x rank count. "
scripts/bench/bench_voronoi_partition_methods.py-254-                        "Uses the production SPMD halo depth, NOT "
scripts/bench/bench_voronoi_partition_methods.py:255:                        "--halo-depth. Expensive: minutes per candidate at "
scripts/bench/bench_voronoi_partition_methods.py-256-                        "subdiv>=8 — run it under batch.")
scripts/bench/bench_voronoi_partition_methods.py:257:    p.add_argument("--lloyd", type=int, default=50,
scripts/bench/bench_voronoi_partition_methods.py-258-                   help="Lloyd relaxation iterations for the mesh. 50 = the "
scripts/bench/bench_voronoi_partition_methods.py-259-                        "production SCVT key; 0 = the LABELLED synthetic "
scripts/bench/bench_voronoi_partition_methods.py:260:                        "scaling mesh. Recorded so a lloyd=0 mesh can never "
scripts/bench/bench_voronoi_partition_methods.py-261-                        "masquerade as a production receipt, and it must "
scripts/bench/bench_voronoi_partition_methods.py-262-                        "match the prewarmed cache key at subdiv>=9.")
scripts/bench/bench_voronoi_partition_methods.py-263-    p.add_argument("--out", type=str,
scripts/bench/bench_voronoi_partition_methods.py-264-                   default="results/a1/voronoi_partition_quality.json")
--
scripts/bench/bench_voronoi_partition_methods.py-276-    from legoesm.grids.voronoi import create_voronoi_mesh
scripts/bench/bench_voronoi_partition_methods.py-277-    from legoesm.parallel.voronoi_partition import resolve_partition_method
scripts/bench/bench_voronoi_partition_methods.py-278-
scripts/bench/bench_voronoi_partition_methods.py-279-    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
scripts/bench/bench_voronoi_partition_methods.py:280:                               lloyd_iterations=args.lloyd)
scripts/bench/bench_voronoi_partition_methods.py-281-    if max(rank_counts) > int(mesh.nCells):
scripts/bench/bench_voronoi_partition_methods.py-282-        raise SystemExit(
scripts/bench/bench_voronoi_partition_methods.py-283-            f"--rank-counts max {max(rank_counts)} exceeds the mesh's "
scripts/bench/bench_voronoi_partition_methods.py-284-            f"{int(mesh.nCells)} cells (empty ranks are meaningless).")
--
scripts/bench/bench_voronoi_partition_methods.py-344-            transport="none",
scripts/bench/bench_voronoi_partition_methods.py-345-            extra={"rank_counts": rank_counts, "methods": methods,
scripts/bench/bench_voronoi_partition_methods.py-346-                   "halo_depth": args.halo_depth,
scripts/bench/bench_voronoi_partition_methods.py-347-                   "schedule_cost": bool(args.schedule_cost),
scripts/bench/bench_voronoi_partition_methods.py:348:                   "lloyd_iterations": args.lloyd},
scripts/bench/bench_voronoi_partition_methods.py-349-        )),
scripts/bench/bench_voronoi_partition_methods.py-350-    }
scripts/bench/bench_voronoi_partition_methods.py-351-    outdir = os.path.dirname(args.out)
scripts/bench/bench_voronoi_partition_methods.py-352-    if outdir:
--
scripts/bench/profile_mpas_ocean.py-1-"""Profile the MPAS-ocean I6/L20 step on a local GPU.
scripts/bench/profile_mpas_ocean.py-2-
scripts/bench/profile_mpas_ocean.py-3-Builds the MPAS Voronoi ocean model exactly as
scripts/bench/profile_mpas_ocean.py-4-``scripts/bench/bench_ocean_gpu_scaling.py`` does (subdivision_level=6,
scripts/bench/profile_mpas_ocean.py:5:lloyd_iterations=5, 20 z* levels, default ``MPASOceanConfig``), then
scripts/bench/profile_mpas_ocean.py-6-breaks per-step wall time into sub-stages using:
scripts/bench/profile_mpas_ocean.py-7-
scripts/bench/profile_mpas_ocean.py-8-* ``time.perf_counter()`` + ``jax.block_until_ready`` around isolated
scripts/bench/profile_mpas_ocean.py-9-  closures for each stage (full segment step via ``lax.scan``, bare
--
scripts/bench/profile_mpas_ocean.py-126-
scripts/bench/profile_mpas_ocean.py-127-def main() -> int:
scripts/bench/profile_mpas_ocean.py-128-    # --- I6/L20 baseline (matches bench_ocean_gpu_scaling). ----------------
scripts/bench/profile_mpas_ocean.py-129-    subdivision_level = 6
scripts/bench/profile_mpas_ocean.py:130:    lloyd_iterations = 5
scripts/bench/profile_mpas_ocean.py-131-    n_levels = 20
scripts/bench/profile_mpas_ocean.py-132-    dt = 600.0
scripts/bench/profile_mpas_ocean.py-133-    precision = "fp64" if os.environ.get("JAX_ENABLE_X64", "0") == "1" else "fp32"
scripts/bench/profile_mpas_ocean.py-134-
--
scripts/bench/profile_mpas_ocean.py-147-    # --- Build mesh + state + model ----------------------------------------
scripts/bench/profile_mpas_ocean.py-148-    t0 = time.perf_counter()
scripts/bench/profile_mpas_ocean.py-149-    mesh = create_voronoi_mesh(
scripts/bench/profile_mpas_ocean.py-150-        subdivision_level=subdivision_level,
scripts/bench/profile_mpas_ocean.py:151:        lloyd_iterations=lloyd_iterations,
scripts/bench/profile_mpas_ocean.py-152-    )
scripts/bench/profile_mpas_ocean.py-153-    z = create_ocean_z_star(n_levels=n_levels)
scripts/bench/profile_mpas_ocean.py-154-    cfg = MPASOceanConfig()
scripts/bench/profile_mpas_ocean.py-155-    model = MPASOceanModel(mesh, z, cfg)
--
tests/distributed/test_voronoi_batched_halo.py-56-
tests/distributed/test_voronoi_batched_halo.py-57-@pytest.fixture(scope="module")
tests/distributed/test_voronoi_batched_halo.py-58-def mesh():
tests/distributed/test_voronoi_batched_halo.py-59-    """Small SCVT mesh (162 cells) — same as test_voronoi_halo."""
tests/distributed/test_voronoi_batched_halo.py:60:    return create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/distributed/test_voronoi_batched_halo.py-61-
tests/distributed/test_voronoi_batched_halo.py-62-
tests/distributed/test_voronoi_batched_halo.py-63-def _build_all(mesh, n_ranks):
tests/distributed/test_voronoi_batched_halo.py-64-    """All ranks' partitions + batched schedules, in one process."""
--
scripts/bench/bench_ocean_gpu_scaling.py-98-    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
scripts/bench/bench_ocean_gpu_scaling.py-99-    from legoesm.ocean.dynamics.ocean_model_mpas import (
scripts/bench/bench_ocean_gpu_scaling.py-100-        MPASOceanModel, MPASOceanConfig,
scripts/bench/bench_ocean_gpu_scaling.py-101-    )
scripts/bench/bench_ocean_gpu_scaling.py:102:    mesh = create_voronoi_mesh(subdivision_level=level, lloyd_iterations=5)
scripts/bench/bench_ocean_gpu_scaling.py-103-    z = create_ocean_z_star(n_levels=OCEAN_NLEV)
scripts/bench/bench_ocean_gpu_scaling.py-104-    cfg = MPASOceanConfig(barotropic_solver=baro_solver)
scripts/bench/bench_ocean_gpu_scaling.py-105-    model = MPASOceanModel(mesh, z, cfg)
scripts/bench/bench_ocean_gpu_scaling.py-106-    state = rest_state_mpas_ocean(mesh, z)
--
tests/distributed/test_voronoi_halo.py-45-
tests/distributed/test_voronoi_halo.py-46-@pytest.fixture(scope="module")
tests/distributed/test_voronoi_halo.py-47-def mesh():
tests/distributed/test_voronoi_halo.py-48-    """Small SCVT mesh for testing (162 cells)."""
tests/distributed/test_voronoi_halo.py:49:    return create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
tests/distributed/test_voronoi_halo.py-50-
tests/distributed/test_voronoi_halo.py-51-
tests/distributed/test_voronoi_halo.py-52-# ========================================================================
tests/distributed/test_voronoi_halo.py-53-# Partition validity
--
tests/distributed/test_voronoi_halo.py-408-    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
tests/distributed/test_voronoi_halo.py-409-    def test_schedule_covers_all_halo_cells(self, mesh, n_ranks):
tests/distributed/test_voronoi_halo.py-410-        """Every halo cell is covered by exactly one ppermute round."""
tests/distributed/test_voronoi_halo.py-411-        from legoesm.parallel.voronoi_partition import (
tests/distributed/test_voronoi_halo.py:412:            reorder_voronoi_for_sharding,
tests/distributed/test_voronoi_halo.py-413-        )
tests/distributed/test_voronoi_halo.py-414-        from legoesm.parallel.sharded_dynamics import (
tests/distributed/test_voronoi_halo.py-415-            _build_voronoi_partition_infra,
tests/distributed/test_voronoi_halo.py-416-            _build_ppermute_schedule,
tests/distributed/test_voronoi_halo.py-417-        )
tests/distributed/test_voronoi_halo.py-418-
tests/distributed/test_voronoi_halo.py:419:        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
tests/distributed/test_voronoi_halo.py-420-        nCells = reordered.nCells
tests/distributed/test_voronoi_halo.py-421-        nEdges = reordered.nEdges
tests/distributed/test_voronoi_halo.py-422-        cells_per = nCells // n_ranks
tests/distributed/test_voronoi_halo.py-423-        edges_per = nEdges // n_ranks
--
tests/distributed/test_voronoi_halo.py-454-    def test_ppermute_halo_matches_simulated_exchange(self, mesh, n_ranks):
tests/distributed/test_voronoi_halo.py-455-        """ppermute schedule produces same halo values as simulated
tests/distributed/test_voronoi_halo.py-456-        exchange."""
tests/distributed/test_voronoi_halo.py-457-        from legoesm.parallel.voronoi_partition import (
tests/distributed/test_voronoi_halo.py:458:            reorder_voronoi_for_sharding,
tests/distributed/test_voronoi_halo.py-459-        )
tests/distributed/test_voronoi_halo.py-460-        from legoesm.parallel.sharded_dynamics import (
tests/distributed/test_voronoi_halo.py-461-            _build_voronoi_partition_infra,
tests/distributed/test_voronoi_halo.py-462-            _build_ppermute_schedule,
tests/distributed/test_voronoi_halo.py-463-        )
tests/distributed/test_voronoi_halo.py-464-
tests/distributed/test_voronoi_halo.py:465:        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
tests/distributed/test_voronoi_halo.py-466-        nCells = reordered.nCells
tests/distributed/test_voronoi_halo.py-467-        cells_per = nCells // n_ranks
tests/distributed/test_voronoi_halo.py-468-        edges_per = reordered.nEdges // n_ranks
tests/distributed/test_voronoi_halo.py-469-
--
tests/distributed/test_voronoi_halo.py-521-    @pytest.mark.parametrize("n_ranks", [2, 4])
tests/distributed/test_voronoi_halo.py-522-    def test_edge_coloring_valid(self, mesh, n_ranks):
tests/distributed/test_voronoi_halo.py-523-        """Each ppermute round has no device appearing as sender twice."""
tests/distributed/test_voronoi_halo.py-524-        from legoesm.parallel.voronoi_partition import (
tests/distributed/test_voronoi_halo.py:525:            reorder_voronoi_for_sharding,
tests/distributed/test_voronoi_halo.py-526-        )
tests/distributed/test_voronoi_halo.py-527-        from legoesm.parallel.sharded_dynamics import (
tests/distributed/test_voronoi_halo.py-528-            _build_voronoi_partition_infra,
tests/distributed/test_voronoi_halo.py-529-            _build_ppermute_schedule,
tests/distributed/test_voronoi_halo.py-530-        )
tests/distributed/test_voronoi_halo.py-531-
tests/distributed/test_voronoi_halo.py:532:        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
tests/distributed/test_voronoi_halo.py-533-        cells_per = reordered.nCells // n_ranks
tests/distributed/test_voronoi_halo.py-534-        edges_per = reordered.nEdges // n_ranks
tests/distributed/test_voronoi_halo.py-535-
tests/distributed/test_voronoi_halo.py-536-        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
--
tests/test_dispatch_hardening.py-208-        ("packages/core/legoesm/parallel/runtime.py", "halo_exchange"),
tests/test_dispatch_hardening.py-209-        ("packages/core/legoesm/parallel/voronoi_mpi.py", "gather_voronoi_field"),
tests/test_dispatch_hardening.py-210-        ("packages/core/legoesm/parallel/voronoi_mpi.py", "initialize_voronoi_mpi"),
tests/test_dispatch_hardening.py-211-        ("packages/core/legoesm/parallel/voronoi_partition.py", "partition_voronoi_mesh"),
tests/test_dispatch_hardening.py:212:        ("packages/core/legoesm/parallel/voronoi_partition.py", "reorder_voronoi_for_sharding"),
tests/test_dispatch_hardening.py-213-        ("packages/core/legoesm/timestepping/dispatch.py", "dispatch_integrator"),
tests/test_dispatch_hardening.py-214-        ("packages/core/legoesm/timestepping/split_explicit.py", "split_explicit_step"),
tests/test_dispatch_hardening.py-215-        ("packages/coupler/legoesm/coupler/coupler.py", "ocean_tile_response"),
tests/test_dispatch_hardening.py-216-        ("packages/coupler/legoesm/coupler/lake/two_layer_lake.py", "step_lake"),
--
tests/unit/test_clubb_scheme.py-1071-    from legoesm.grids.vertical import create_sigma_coordinate
tests/unit/test_clubb_scheme.py-1072-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_clubb_scheme.py-1073-
tests/unit/test_clubb_scheme.py-1074-    nlev = 10
tests/unit/test_clubb_scheme.py:1075:    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
tests/unit/test_clubb_scheme.py-1076-    sigma = create_sigma_coordinate(nlev)
tests/unit/test_clubb_scheme.py-1077-    nC, nE = mesh.nCells, mesh.nEdges
tests/unit/test_clubb_scheme.py-1078-    # Sheared (2→12 m/s) + per-edge structured edge-normal wind so the Perot
tests/unit/test_clubb_scheme.py-1079-    # reconstruction yields non-zero, spatially-varying cell winds.
--
tests/unit/test_clubb_scheme.py-1152-    from legoesm.grids.vertical import create_sigma_coordinate
tests/unit/test_clubb_scheme.py-1153-    from legoesm.grids.voronoi import create_voronoi_mesh
tests/unit/test_clubb_scheme.py-1154-
tests/unit/test_clubb_scheme.py-1155-    nlev = 10
tests/unit/test_clubb_scheme.py:1156:    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
tests/unit/test_clubb_scheme.py-1157-    sigma = create_sigma_coordinate(nlev)
tests/unit/test_clubb_scheme.py-1158-    nC, nE = mesh.nCells, mesh.nEdges
tests/unit/test_clubb_scheme.py-1159-    rng = np.random.default_rng(0)
tests/unit/test_clubb_scheme.py-1160-    u_edge = jnp.asarray(np.linspace(2.0, 10.0, nlev)[None, :]
--
packages/core/legoesm/parallel/__init__.py-143-    DeviceConfig,
packages/core/legoesm/parallel/__init__.py-144-    create_device_mesh,
packages/core/legoesm/parallel/__init__.py-145-    create_latlon_mesh,
packages/core/legoesm/parallel/__init__.py-146-    create_level_mesh,
packages/core/legoesm/parallel/__init__.py:147:    create_voronoi_device_mesh,
packages/core/legoesm/parallel/__init__.py-148-    get_active_config,
packages/core/legoesm/parallel/__init__.py-149-    replicate_pytree,
packages/core/legoesm/parallel/__init__.py-150-    shard_latlon,
packages/core/legoesm/parallel/__init__.py-151-    shard_levels,
--
packages/core/legoesm/parallel/__init__.py-178-    create_output_shardings,
packages/core/legoesm/parallel/__init__.py-179-    gather_voronoi_state_spmd,
packages/core/legoesm/parallel/__init__.py-180-    make_face_halo_exchange,
packages/core/legoesm/parallel/__init__.py-181-    make_sharded_step,
packages/core/legoesm/parallel/__init__.py:182:    make_voronoi_sharded_step,
packages/core/legoesm/parallel/__init__.py-183-    sharded_integrate,
packages/core/legoesm/parallel/__init__.py-184-    sharded_integrate_scan,
packages/core/legoesm/parallel/__init__.py-185-    sharded_step_with_halo,
packages/core/legoesm/parallel/__init__.py-186-)
--
packages/core/legoesm/parallel/__init__.py-197-    hilbert_cell_keys,
packages/core/legoesm/parallel/__init__.py-198-    partition_cells_geometric,
packages/core/legoesm/parallel/__init__.py-199-    partition_cells_sfc,
packages/core/legoesm/parallel/__init__.py-200-    partition_voronoi_mesh,
packages/core/legoesm/parallel/__init__.py:201:    reorder_voronoi_for_sharding,
packages/core/legoesm/parallel/__init__.py-202-    resolve_partition_method,
packages/core/legoesm/parallel/__init__.py-203-    scatter_to_local,
packages/core/legoesm/parallel/__init__.py-204-)
packages/core/legoesm/parallel/__init__.py-205-
--
packages/core/legoesm/parallel/__init__.py-213-    "DeviceConfig",
packages/core/legoesm/parallel/__init__.py-214-    "create_device_mesh",
packages/core/legoesm/parallel/__init__.py-215-    "create_latlon_mesh",
packages/core/legoesm/parallel/__init__.py-216-    "create_level_mesh",
packages/core/legoesm/parallel/__init__.py:217:    "create_voronoi_device_mesh",
packages/core/legoesm/parallel/__init__.py-218-    "get_active_config",
packages/core/legoesm/parallel/__init__.py-219-    "replicate_pytree",
packages/core/legoesm/parallel/__init__.py-220-    "shard_pytree",
packages/core/legoesm/parallel/__init__.py-221-    "shard_latlon",
--
packages/core/legoesm/parallel/__init__.py-228-    "hilbert_cell_keys",
packages/core/legoesm/parallel/__init__.py-229-    "partition_voronoi_mesh",
packages/core/legoesm/parallel/__init__.py-230-    "build_local_mesh",
packages/core/legoesm/parallel/__init__.py-231-    "scatter_to_local",
packages/core/legoesm/parallel/__init__.py:232:    "reorder_voronoi_for_sharding",
packages/core/legoesm/parallel/__init__.py-233-    "resolve_partition_method",
packages/core/legoesm/parallel/__init__.py-234-    "VoronoiHaloExchange",
packages/core/legoesm/parallel/__init__.py-235-    "exchange_local_simulated",
packages/core/legoesm/parallel/__init__.py-236-    # Ensemble parallelism
--
packages/core/legoesm/parallel/mesh.py-644-# ==============================================================================
packages/core/legoesm/parallel/mesh.py-645-# Voronoi (icosahedral) mesh parallelism
packages/core/legoesm/parallel/mesh.py-646-# ==============================================================================
packages/core/legoesm/parallel/mesh.py-647-
packages/core/legoesm/parallel/mesh.py:648:def create_voronoi_device_mesh(
packages/core/legoesm/parallel/mesh.py-649-    nCells: int,
packages/core/legoesm/parallel/mesh.py-650-    nEdges: int,
packages/core/legoesm/parallel/mesh.py-651-    nVertices: int,
packages/core/legoesm/parallel/mesh.py-652-    n_devices: int | str = "auto",
--
packages/core/legoesm/parallel/mesh.py-656-    """Create a JAX device mesh for Voronoi (MPAS) mesh parallelism.
packages/core/legoesm/parallel/mesh.py-657-
packages/core/legoesm/parallel/mesh.py-658-    Shards cell- and edge-centered arrays along axis 0 across devices.
packages/core/legoesm/parallel/mesh.py-659-    For best performance, reorder the mesh with
packages/core/legoesm/parallel/mesh.py:660:    :func:`legoesm.parallel.voronoi_partition.reorder_voronoi_for_sharding`
packages/core/legoesm/parallel/mesh.py-661-    before sharding so that each device gets a spatially contiguous cell
packages/core/legoesm/parallel/mesh.py-662-    cluster.
packages/core/legoesm/parallel/mesh.py-663-
packages/core/legoesm/parallel/mesh.py-664-    Parameters
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-10-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-11-#SBATCH --output=mpas_s8_l0.%j.log
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-12-# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:14:# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:21:#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-22-#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-23-#             matched cells/GPU stay well above 1 (prior draft saw
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-24-#             1.80/1.35/1.41 on the CONFOUNDED pairs)
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-25-#   REFUTE  : ratios collapse toward ~1.0 -> the draft's "term" was the
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-40-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-41-rc=0
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-42-for NP in 8 16 32; do
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-43-  NODES=$(( NP / 4 ))
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:44:  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-45-  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-46-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-47-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-48-      --multicontroller --n-devices "$NP" \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:49:      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-50-      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-53-echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
--
packages/core/legoesm/parallel/voronoi_partition.py-1023-        return "sfc"
packages/core/legoesm/parallel/voronoi_partition.py-1024-    return resolve_partition_method(method)
packages/core/legoesm/parallel/voronoi_partition.py-1025-
packages/core/legoesm/parallel/voronoi_partition.py-1026-
packages/core/legoesm/parallel/voronoi_partition.py:1027:def reorder_voronoi_for_sharding(
packages/core/legoesm/parallel/voronoi_partition.py-1028-    mesh: VoronoiMesh,
packages/core/legoesm/parallel/voronoi_partition.py-1029-    n_devices: int,
packages/core/legoesm/parallel/voronoi_partition.py-1030-    *,
packages/core/legoesm/parallel/voronoi_partition.py-1031-    method: str = "auto",
--
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-40-#     schedule there and the number is counterfactual.  Only meshes big
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-41-#     enough to keep cells/device above the threshold answer the question.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-42-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-43-# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:44:# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-45-#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-46-#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-47-# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-48-# A scan that misses the known answer is a broken instrument, and its s10
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-49-# row must not be believed.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-50-#
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:51:# lloyd=0 throughout: it is what the reference census used AND the cache key
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-52-# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-53-# synthetic scaling mesh, recorded in every row's metadata so it can never be
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-54-# read back as a production SCVT receipt.
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-55-#
--
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-82-"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-83-  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-84-
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-85-run_arm () {  # $1=level  $2=rank-counts  $3=label
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:86:  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-87-  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-88-  "$PY" "$BENCH" \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:89:      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-90-      --methods geometric,sfc,metis --schedule-cost \
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-91-      --out "$OUT/schedule_cost_s$1.json"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-92-  echo "[scan] arm $3 exit=$?"
scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-93-  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-8-#SBATCH --exclusive
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-10-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-11-#SBATCH --output=mpas_s10_128.%j.log
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:12:# HUNDREDS-OF-GPUS MPAS, rung 1: subdiv-10 (10.49M cells, lloyd=0
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-13-# synthetic, prewarmed job 26628074) at 128 GPUs = 81.9k cells/GPU —
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-14-# the SAME near-matched tile as s8@np8 (6.58 ms) and s9@np32 (12.47 ms),
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-15-# extending the 4x-devices weak series a third rung.
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-16-# Falsifiability, written BEFORE submit:
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-32-JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-33-    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-34-  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-35-    --multicontroller --n-devices 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:36:    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-37-    --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-38-    --out "$OUTDIR/np128.jsonl" || { echo "np128 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-39-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-40-import json,math,sys
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-190-    [ "$rc" -ne 0 ] && { echo "LANE D FAILED rc=$rc"; rc_all=$rc; }
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-191-fi
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-192-
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-193-# Lane E: icosahedral/MPAS multicontroller (6 procs = 2 nodes x 3 GPUs) —
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:194:# cell-partition reorder + ppermute halo (make_voronoi_sharded_step) over
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-195-# jax.distributed. np=6: nCells = 10*4^L+2 admits 1/2/3/6 even splits at
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-196-# every level. Smoke gate (parity+conservation) first, then the timed case.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-197-# Gate: tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-198-if [ "$RUN_MPAS" = "1" ]; then
--
packages/core/legoesm/parallel/sharded_dynamics.py-107-            "This sharded step wrapper does not thread the PhysicsState "
packages/core/legoesm/parallel/sharded_dynamics.py-108-            "carry, so the configured stateful physics would silently "
packages/core/legoesm/parallel/sharded_dynamics.py-109-            "reseed every step (issue #405/#413).  Use a diagnostic "
packages/core/legoesm/parallel/sharded_dynamics.py-110-            "scheme, the ModelDriver loops / MPAS step, or "
packages/core/legoesm/parallel/sharded_dynamics.py:111:            "make_voronoi_sharded_step(return_phys_state=True), which "
packages/core/legoesm/parallel/sharded_dynamics.py-112-            "thread the carry."
packages/core/legoesm/parallel/sharded_dynamics.py-113-        )
packages/core/legoesm/parallel/sharded_dynamics.py-114-
packages/core/legoesm/parallel/sharded_dynamics.py-115-
--
packages/core/legoesm/parallel/sharded_dynamics.py-1243-#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
packages/core/legoesm/parallel/sharded_dynamics.py-1244-#: consumed by both the production step factory and ``spmd_schedule_cost``:
packages/core/legoesm/parallel/sharded_dynamics.py-1245-#: a score computed at a different depth describes a different comm graph, and
packages/core/legoesm/parallel/sharded_dynamics.py-1246-#: two independently hardcoded 3s let production drift unnoticed.
packages/core/legoesm/parallel/sharded_dynamics.py:1247:SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py-1248-
packages/core/legoesm/parallel/sharded_dynamics.py-1249-
packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
packages/core/legoesm/parallel/sharded_dynamics.py:1251:                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py-1252-                       ppermute_cells_per_device_threshold=2_000):
packages/core/legoesm/parallel/sharded_dynamics.py-1253-    """How much halo communication one ownership choice costs, computed offline.
packages/core/legoesm/parallel/sharded_dynamics.py-1254-
packages/core/legoesm/parallel/sharded_dynamics.py-1255-    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
--
packages/core/legoesm/parallel/sharded_dynamics.py-1281-      see the returned ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py-1282-
packages/core/legoesm/parallel/sharded_dynamics.py-1283-    MESH STATE -- the one thing that silently invalidates the score
packages/core/legoesm/parallel/sharded_dynamics.py-1284-    --------------------------------------------------------------
packages/core/legoesm/parallel/sharded_dynamics.py:1285:    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
packages/core/legoesm/parallel/sharded_dynamics.py-1286-    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
packages/core/legoesm/parallel/sharded_dynamics.py:1287:    ONCE for a ``reorder_target`` device count and then runs at a possibly
packages/core/legoesm/parallel/sharded_dynamics.py-1288-    DIFFERENT device count.  So pass what you actually have:
packages/core/legoesm/parallel/sharded_dynamics.py-1289-
packages/core/legoesm/parallel/sharded_dynamics.py-1290-    * raw mesh, scoring a run at ``n_dev``: defaults are right.
packages/core/legoesm/parallel/sharded_dynamics.py-1291-    * raw mesh, but the run reorders for a different target: pass
packages/core/legoesm/parallel/sharded_dynamics.py:1292:      ``reorder_target=<that target>``; the split is built for the target and
packages/core/legoesm/parallel/sharded_dynamics.py-1293-      scored at ``n_dev``.
packages/core/legoesm/parallel/sharded_dynamics.py-1294-    * already-reordered mesh (what production holds): pass
packages/core/legoesm/parallel/sharded_dynamics.py-1295-      ``already_reordered=True``; ``method`` is then ignored and reported as
packages/core/legoesm/parallel/sharded_dynamics.py-1296-      ``"pre-reordered"``, because the ownership is already baked in.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1301-    n_dev : int
packages/core/legoesm/parallel/sharded_dynamics.py-1302-        Device count the run uses.  Must be >= 1.
packages/core/legoesm/parallel/sharded_dynamics.py-1303-    method : str
packages/core/legoesm/parallel/sharded_dynamics.py-1304-        Ownership for the reorder; ignored when *already_reordered*.
packages/core/legoesm/parallel/sharded_dynamics.py:1305:    reorder_target : int | None
packages/core/legoesm/parallel/sharded_dynamics.py-1306-        Device count the reorder targets, when it differs from *n_dev*.
packages/core/legoesm/parallel/sharded_dynamics.py-1307-    already_reordered : bool
packages/core/legoesm/parallel/sharded_dynamics.py-1308-    halo_depth : int
packages/core/legoesm/parallel/sharded_dynamics.py-1309-        Must match production (3) or the graph is a different graph.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1324-    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
packages/core/legoesm/parallel/sharded_dynamics.py-1325-    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
packages/core/legoesm/parallel/sharded_dynamics.py-1326-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1327-    from legoesm.parallel.voronoi_partition import (
packages/core/legoesm/parallel/sharded_dynamics.py:1328:        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
packages/core/legoesm/parallel/sharded_dynamics.py-1329-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1330-
packages/core/legoesm/parallel/sharded_dynamics.py-1331-    if int(n_dev) != n_dev or int(n_dev) < 1:
packages/core/legoesm/parallel/sharded_dynamics.py-1332-        # int() would silently truncate 3.9 -> 3 and score the wrong split.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1334-            f"spmd_schedule_cost: n_dev must be an integer >= 1, got {n_dev!r}")
packages/core/legoesm/parallel/sharded_dynamics.py-1335-    n_dev = int(n_dev)
packages/core/legoesm/parallel/sharded_dynamics.py-1336-
packages/core/legoesm/parallel/sharded_dynamics.py-1337-    if already_reordered:
packages/core/legoesm/parallel/sharded_dynamics.py:1338:        if reorder_target is not None:
packages/core/legoesm/parallel/sharded_dynamics.py-1339-            raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py:1340:                "spmd_schedule_cost: reorder_target is meaningless with "
packages/core/legoesm/parallel/sharded_dynamics.py-1341-                "already_reordered=True — the ownership is already baked into "
packages/core/legoesm/parallel/sharded_dynamics.py-1342-                "the mesh.")
packages/core/legoesm/parallel/sharded_dynamics.py-1343-        prepared, resolved = mesh, "pre-reordered"
packages/core/legoesm/parallel/sharded_dynamics.py-1344-    else:
packages/core/legoesm/parallel/sharded_dynamics.py:1345:        target = n_dev if reorder_target is None else int(reorder_target)
packages/core/legoesm/parallel/sharded_dynamics.py:1346:        prepared = reorder_voronoi_for_sharding(mesh, target, method=method)
packages/core/legoesm/parallel/sharded_dynamics.py-1347-        # Report the CONCRETE ownership: "auto" hides which partitioner ran.
packages/core/legoesm/parallel/mesh.py-62-    tiling : tuple[int, int]
packages/core/legoesm/parallel/mesh.py-63-        Sub-face tile grid ``(tx, ty)``.  ``(1, 1)`` means face-only
packages/core/legoesm/parallel/mesh.py-64-        sharding (no sub-face tiling).
packages/core/legoesm/parallel/mesh.py-65-    grid_type : str
packages/core/legoesm/parallel/mesh.py-66-        ``"cubed_sphere"``, ``"latlon"``, ``"spectral"``, or ``"voronoi"``.
packages/core/legoesm/parallel/mesh.py:67:    voronoi_dims : tuple[int, int, int] or None
packages/core/legoesm/parallel/mesh.py:68:        ``(nCells, nEdges, nVertices)`` for voronoi grids.  ``None`` for
packages/core/legoesm/parallel/mesh.py-69-        other grid types.  Used by :func:`shard_pytree` to identify which
packages/core/legoesm/parallel/mesh.py-70-        arrays to shard along axis 0.
packages/core/legoesm/parallel/mesh.py-71-    """
packages/core/legoesm/parallel/mesh.py-72-    mesh: Mesh | None
packages/core/legoesm/parallel/mesh.py-73-    face_sharding: NamedSharding | None
--
packages/core/legoesm/parallel/mesh.py-75-    n_devices: int
packages/core/legoesm/parallel/mesh.py-76-    backend: str
packages/core/legoesm/parallel/mesh.py-77-    is_distributed: bool
packages/core/legoesm/parallel/mesh.py-78-    tiling: tuple[int, int] = (1, 1)
packages/core/legoesm/parallel/mesh.py-79-    grid_type: str = "cubed_sphere"
packages/core/legoesm/parallel/mesh.py:80:    voronoi_dims: tuple[int, int, int] | None = None
packages/core/legoesm/parallel/mesh.py-81-
packages/core/legoesm/parallel/mesh.py-82-
packages/core/legoesm/parallel/mesh.py-83-# Singleton — set once at startup, queried by the rest of the code.
packages/core/legoesm/parallel/mesh.py-84-_active_config: DeviceConfig | None = None
packages/core/legoesm/parallel/mesh.py-85-
--
packages/core/legoesm/parallel/mesh.py-643-
packages/core/legoesm/parallel/mesh.py-644-# ==============================================================================
packages/core/legoesm/parallel/mesh.py-645-# Voronoi (icosahedral) mesh parallelism
packages/core/legoesm/parallel/mesh.py-646-# ==============================================================================
packages/core/legoesm/parallel/mesh.py-647-
packages/core/legoesm/parallel/mesh.py:648:def create_voronoi_device_mesh(
packages/core/legoesm/parallel/mesh.py-649-    nCells: int,
packages/core/legoesm/parallel/mesh.py-650-    nEdges: int,
packages/core/legoesm/parallel/mesh.py-651-    nVertices: int,
packages/core/legoesm/parallel/mesh.py-652-    n_devices: int | str = "auto",
packages/core/legoesm/parallel/mesh.py-653-    backend: str | None = None,
packages/core/legoesm/parallel/mesh.py-654-    devices: Sequence | None = None,
packages/core/legoesm/parallel/mesh.py-655-) -> DeviceConfig:
packages/core/legoesm/parallel/mesh.py-656-    """Create a JAX device mesh for Voronoi (MPAS) mesh parallelism.
packages/core/legoesm/parallel/mesh.py-657-
packages/core/legoesm/parallel/mesh.py-658-    Shards cell- and edge-centered arrays along axis 0 across devices.
packages/core/legoesm/parallel/mesh.py:659:    For best performance, reorder the mesh with
packages/core/legoesm/parallel/mesh.py:660:    :func:`legoesm.parallel.voronoi_partition.reorder_voronoi_for_sharding`
packages/core/legoesm/parallel/mesh.py-661-    before sharding so that each device gets a spatially contiguous cell
packages/core/legoesm/parallel/mesh.py-662-    cluster.
packages/core/legoesm/parallel/mesh.py-663-
packages/core/legoesm/parallel/mesh.py-664-    Parameters
packages/core/legoesm/parallel/mesh.py-665-    ----------
--
packages/core/legoesm/parallel/mesh.py-677-        Optional explicit device list.
packages/core/legoesm/parallel/mesh.py-678-
packages/core/legoesm/parallel/mesh.py-679-    Returns
packages/core/legoesm/parallel/mesh.py-680-    -------
packages/core/legoesm/parallel/mesh.py-681-    DeviceConfig
packages/core/legoesm/parallel/mesh.py:682:        Config with ``grid_type="voronoi"`` and ``voronoi_dims`` set.
packages/core/legoesm/parallel/mesh.py-683-    """
packages/core/legoesm/parallel/mesh.py-684-    global _active_config
packages/core/legoesm/parallel/mesh.py-685-
packages/core/legoesm/parallel/mesh.py-686-    if devices is not None:
packages/core/legoesm/parallel/mesh.py-687-        devices = list(devices)
--
packages/core/legoesm/parallel/mesh.py-710-            n_devices=1,
packages/core/legoesm/parallel/mesh.py-711-            backend=backend_name,
packages/core/legoesm/parallel/mesh.py-712-            is_distributed=False,
packages/core/legoesm/parallel/mesh.py-713-            tiling=(1, 1),
packages/core/legoesm/parallel/mesh.py-714-            grid_type="voronoi",
packages/core/legoesm/parallel/mesh.py:715:            voronoi_dims=(nCells, nEdges, nVertices),
packages/core/legoesm/parallel/mesh.py-716-        )
packages/core/legoesm/parallel/mesh.py-717-        _active_config = config
packages/core/legoesm/parallel/mesh.py-718-        return config
packages/core/legoesm/parallel/mesh.py-719-
packages/core/legoesm/parallel/mesh.py-720-    selected = devices[:n_dev]
--
packages/core/legoesm/parallel/mesh.py-729-        n_devices=n_dev,
packages/core/legoesm/parallel/mesh.py-730-        backend=backend_name,
packages/core/legoesm/parallel/mesh.py-731-        is_distributed=False,
packages/core/legoesm/parallel/mesh.py-732-        tiling=(1, 1),
packages/core/legoesm/parallel/mesh.py-733-        grid_type="voronoi",
packages/core/legoesm/parallel/mesh.py:734:        voronoi_dims=(nCells, nEdges, nVertices),
packages/core/legoesm/parallel/mesh.py-735-    )
packages/core/legoesm/parallel/mesh.py-736-    _active_config = config
packages/core/legoesm/parallel/mesh.py-737-    logger.info(
packages/core/legoesm/parallel/mesh.py-738-        "legoESM: %d-device voronoi mesh on %s "
packages/core/legoesm/parallel/mesh.py:739:        "(nCells=%d, nEdges=%d, nVertices=%d)",
packages/core/legoesm/parallel/mesh.py:740:        n_dev, backend_name, nCells, nEdges, nVertices,
packages/core/legoesm/parallel/mesh.py-741-    )
packages/core/legoesm/parallel/mesh.py-742-    return config
packages/core/legoesm/parallel/mesh.py-743-
packages/core/legoesm/parallel/mesh.py-744-
packages/core/legoesm/parallel/mesh.py-745-# ==============================================================================
--
packages/core/legoesm/parallel/mesh.py-864-            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-865-
packages/core/legoesm/parallel/mesh.py-866-        elif config.grid_type == "voronoi":
packages/core/legoesm/parallel/mesh.py-867-            # Voronoi state arrays: shard cell- and edge-centered arrays
packages/core/legoesm/parallel/mesh.py-868-            # along axis 0.  Vertex arrays and small arrays are replicated.
packages/core/legoesm/parallel/mesh.py:869:            if config.voronoi_dims is not None and leaf.ndim >= 1:
packages/core/legoesm/parallel/mesh.py:870:                nCells, nEdges, _nVerts = config.voronoi_dims
packages/core/legoesm/parallel/mesh.py:871:                if leaf.shape[0] in (nCells, nEdges):
packages/core/legoesm/parallel/mesh.py-872-                    return multiprocess_safe_device_put(leaf, config.face_sharding)
packages/core/legoesm/parallel/mesh.py-873-            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-874-
packages/core/legoesm/parallel/mesh.py-875-        # Default: replicate
packages/core/legoesm/parallel/mesh.py-876-        return multiprocess_safe_device_put(leaf, config.replicated_sharding)
--
packages/core/legoesm/io/cmor_table_loader.py-148-}
packages/core/legoesm/io/cmor_table_loader.py-149-# Any ``plevNN`` / ``plevNNh`` token collapses to the ``plev`` axis.
packages/core/legoesm/io/cmor_table_loader.py-150-_PLEV_RE = re.compile(r"^plev\d+h?$")
packages/core/legoesm/io/cmor_table_loader.py-151-
packages/core/legoesm/io/cmor_table_loader.py-152-# Ordering of the internal dimension tuple.  Rank drives the sort so a
packages/core/legoesm/io/cmor_table_loader.py:153:# reordering of the official token string cannot change our layout.
packages/core/legoesm/io/cmor_table_loader.py-154-_DIM_ORDER = {"time": 0, "basin": 1, "plev": 2, "lev": 2, "depth": 2, "lat": 3, "lon": 4}
packages/core/legoesm/io/cmor_table_loader.py-155-
packages/core/legoesm/io/cmor_table_loader.py-156-# KNOWN DEVIATIONS from the official table, kept because correcting them
packages/core/legoesm/io/cmor_table_loader.py-157-# would change the DATA rather than the metadata.  Each one is a
packages/core/legoesm/io/cmor_table_loader.py-158-# publication blocker for the affected variable and is asserted, by exact
--
packages/core/legoesm/grids/voronoi.py-759-            if c >= 0:
packages/core/legoesm/grids/voronoi.py-760-                areaCell[c] += kiteAreasOnVertex[k, v]
packages/core/legoesm/grids/voronoi.py-761-
packages/core/legoesm/grids/voronoi.py-762-    # --- Edge sign arrays ---
packages/core/legoesm/grids/voronoi.py-763-    edgeSignOnCell = _compute_edge_sign_on_cell(
packages/core/legoesm/grids/voronoi.py:764:        nCells, maxEdges, nEdgesOnCell_arr, edgesOnCell, cellsOnEdge)
packages/core/legoesm/grids/voronoi.py-765-    edgeSignOnVertex = _compute_edge_sign_on_vertex(
packages/core/legoesm/grids/voronoi.py-766-        nVertices, vertexDegree, edgesOnVertex, cellsOnVertex, cellsOnEdge,
packages/core/legoesm/grids/voronoi.py-767-        vertex_xyz, cell_xyz, periodic_L_rad=_L)
packages/core/legoesm/grids/voronoi.py-768-
packages/core/legoesm/grids/voronoi.py-769-    # --- edgesOnEdge and weightsOnEdge ---
--
packages/core/legoesm/grids/voronoi.py-797-
packages/core/legoesm/grids/voronoi.py-798-    def to_jax_i(arr):
packages/core/legoesm/grids/voronoi.py-799-        return jnp.array(arr, dtype=jnp.int32)
packages/core/legoesm/grids/voronoi.py-800-
packages/core/legoesm/grids/voronoi.py-801-    return VoronoiMesh(
packages/core/legoesm/grids/voronoi.py:802:        nCells=nCells, nEdges=nEdges, nVertices=nVertices,
packages/core/legoesm/grids/voronoi.py-803-        maxEdges=maxEdges, vertexDegree=vertexDegree, radius=radius,
packages/core/legoesm/grids/voronoi.py-804-        latCell=to_jax_f(latCell), lonCell=to_jax_f(lonCell),
packages/core/legoesm/grids/voronoi.py-805-        xCell=to_jax_f(cell_xyz[:, 0] * radius),
packages/core/legoesm/grids/voronoi.py-806-        yCell=to_jax_f(cell_xyz[:, 1] * radius),
packages/core/legoesm/grids/voronoi.py-807-        zCell=to_jax_f(cell_xyz[:, 2] * radius),
--
packages/core/legoesm/grids/voronoi.py-883-            kiteAreas[k, v] = a1 + a2
packages/core/legoesm/grids/voronoi.py-884-
packages/core/legoesm/grids/voronoi.py-885-    return kiteAreas
packages/core/legoesm/grids/voronoi.py-886-
packages/core/legoesm/grids/voronoi.py-887-
packages/core/legoesm/grids/voronoi.py:888:def _compute_edge_sign_on_cell(nCells, maxEdges, nEdgesOnCell, edgesOnCell,
packages/core/legoesm/grids/voronoi.py-889-                                cellsOnEdge):
packages/core/legoesm/grids/voronoi.py-890-    """Compute edgeSignOnCell: +1 if edge normal points outward from cell."""
packages/core/legoesm/grids/voronoi.py-891-    edgeSignOnCell = np.zeros((maxEdges, nCells), dtype=np.float64)
packages/core/legoesm/grids/voronoi.py-892-    for c in range(nCells):
packages/core/legoesm/grids/voronoi.py-893-        n = nEdgesOnCell[c]
--
packages/core/legoesm/grids/voronoi.py-1960-        _fdtype = get_policy().storage
packages/core/legoesm/grids/voronoi.py-1961-    except Exception:
packages/core/legoesm/grids/voronoi.py-1962-        _fdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
packages/core/legoesm/grids/voronoi.py-1963-
packages/core/legoesm/grids/voronoi.py-1964-    mesh_data = {
packages/core/legoesm/grids/voronoi.py:1965:        'nCells': nCells, 'nEdges': nEdges, 'nVertices': nVertices,
packages/core/legoesm/grids/voronoi.py-1966-        'maxEdges': maxEdges, 'vertexDegree': vertexDegree, 'radius': radius,
packages/core/legoesm/grids/voronoi.py-1967-    }
packages/core/legoesm/grids/voronoi.py-1968-
packages/core/legoesm/grids/voronoi.py-1969-    # Coordinates
packages/core/legoesm/grids/voronoi.py-1970-    for name in ['latCell', 'lonCell', 'latEdge', 'lonEdge',
--
packages/core/legoesm/grids/topography.py-772-
packages/core/legoesm/grids/topography.py-773-    Parameters
packages/core/legoesm/grids/topography.py-774-    ----------
packages/core/legoesm/grids/topography.py-775-    phis : (nCells,) surface geopotential [m^2/s^2].
packages/core/legoesm/grids/topography.py-776-    cells_on_cell : (maxEdges, nCells) ``VoronoiMesh.cellsOnCell``.
packages/core/legoesm/grids/topography.py:777:    n_edges_on_cell : (nCells,) ``VoronoiMesh.nEdgesOnCell``.
packages/core/legoesm/grids/topography.py-778-    smoothing_passes : int
packages/core/legoesm/grids/topography.py-779-        Number of Laplacian passes.  Default matches
packages/core/legoesm/grids/topography.py-780-        ``TopographyConfig.smoothing_passes = 4`` and the cubed-sphere path.
packages/core/legoesm/grids/topography.py-781-
packages/core/legoesm/grids/topography.py-782-    Returns
--
packages/core/legoesm/parallel/__init__.py-196-    build_local_mesh,
packages/core/legoesm/parallel/__init__.py-197-    hilbert_cell_keys,
packages/core/legoesm/parallel/__init__.py-198-    partition_cells_geometric,
packages/core/legoesm/parallel/__init__.py-199-    partition_cells_sfc,
packages/core/legoesm/parallel/__init__.py-200-    partition_voronoi_mesh,
packages/core/legoesm/parallel/__init__.py:201:    reorder_voronoi_for_sharding,
packages/core/legoesm/parallel/__init__.py-202-    resolve_partition_method,
packages/core/legoesm/parallel/__init__.py-203-    scatter_to_local,
packages/core/legoesm/parallel/__init__.py-204-)
packages/core/legoesm/parallel/__init__.py-205-
packages/core/legoesm/parallel/__init__.py-206-__all__ = [
--
packages/core/legoesm/parallel/__init__.py-227-    "partition_cells_sfc",
packages/core/legoesm/parallel/__init__.py-228-    "hilbert_cell_keys",
packages/core/legoesm/parallel/__init__.py-229-    "partition_voronoi_mesh",
packages/core/legoesm/parallel/__init__.py-230-    "build_local_mesh",
packages/core/legoesm/parallel/__init__.py-231-    "scatter_to_local",
packages/core/legoesm/parallel/__init__.py:232:    "reorder_voronoi_for_sharding",
packages/core/legoesm/parallel/__init__.py-233-    "resolve_partition_method",
packages/core/legoesm/parallel/__init__.py-234-    "VoronoiHaloExchange",
packages/core/legoesm/parallel/__init__.py-235-    "exchange_local_simulated",
packages/core/legoesm/parallel/__init__.py-236-    # Ensemble parallelism
packages/core/legoesm/parallel/__init__.py-237-    "stack_states",
--
packages/core/legoesm/grids/latlon.py-895-    # Exactly-uniform dy_deg → delegate to the canonical uniform
packages/core/legoesm/grids/latlon.py-896-    # builder: BIT-identical output to create_regional_latlon_grid
packages/core/legoesm/grids/latlon.py-897-    # (linspace centre placement + product-form areas), and downstream
packages/core/legoesm/grids/latlon.py-898-    # consumers see the one well-trodden uniform-grid object.  The
packages/core/legoesm/grids/latlon.py-899-    # stretched path below agrees with it only to float64 round-off
packages/core/legoesm/grids/latlon.py:900:    # (cumsum + u-centred recursion reorder the float ops), so the
packages/core/legoesm/grids/latlon.py-901-    # branch is continuous at the ~ULP level for near-uniform input.
packages/core/legoesm/grids/latlon.py-902-    if bool(np.all(d_int == d_int[0])):
packages/core/legoesm/grids/latlon.py-903-        return create_regional_latlon_grid(
packages/core/legoesm/grids/latlon.py-904-            n_lat=int(d_int.size),
packages/core/legoesm/grids/latlon.py-905-            n_lon=n_lon,
--
packages/core/legoesm/grids/gaussian.py-1408-    n_sh = int(Pnm.shape[1])
packages/core/legoesm/grids/gaussian.py-1409-    # Max modes for any single m: (m..n_max) => at most n_max+1 (the m=0 column).
packages/core/legoesm/grids/gaussian.py-1410-    n_pad = n_max + 1
packages/core/legoesm/grids/gaussian.py-1411-
packages/core/legoesm/grids/gaussian.py-1412-    # Cache key: the scalar shape signature PLUS a byte-exact content hash of
packages/core/legoesm/grids/gaussian.py:1413:    # every operator the inverse depends on, so a custom/modified/reordered
packages/core/legoesm/grids/gaussian.py-1414-    # grid (same scalars, different Pnm/Hnm/lap/ms/ls) does NOT alias another
packages/core/legoesm/grids/gaussian.py-1415-    # grid's inverse.
packages/core/legoesm/grids/gaussian.py-1416-    _h = hashlib.blake2b(digest_size=16)
packages/core/legoesm/grids/gaussian.py-1417-    for _arr in (Pnm, Hnm, lap, ms, ls):
packages/core/legoesm/grids/gaussian.py-1418-        _c = np.ascontiguousarray(_arr)
--
packages/core/legoesm/parallel/mesh.py-62-    tiling : tuple[int, int]
packages/core/legoesm/parallel/mesh.py-63-        Sub-face tile grid ``(tx, ty)``.  ``(1, 1)`` means face-only
packages/core/legoesm/parallel/mesh.py-64-        sharding (no sub-face tiling).
packages/core/legoesm/parallel/mesh.py-65-    grid_type : str
packages/core/legoesm/parallel/mesh.py-66-        ``"cubed_sphere"``, ``"latlon"``, ``"spectral"``, or ``"voronoi"``.
packages/core/legoesm/parallel/mesh.py:67:    voronoi_dims : tuple[int, int, int] or None
packages/core/legoesm/parallel/mesh.py:68:        ``(nCells, nEdges, nVertices)`` for voronoi grids.  ``None`` for
packages/core/legoesm/parallel/mesh.py-69-        other grid types.  Used by :func:`shard_pytree` to identify which
packages/core/legoesm/parallel/mesh.py-70-        arrays to shard along axis 0.
packages/core/legoesm/parallel/mesh.py-71-    """
packages/core/legoesm/parallel/mesh.py-72-    mesh: Mesh | None
packages/core/legoesm/parallel/mesh.py-73-    face_sharding: NamedSharding | None
--
packages/core/legoesm/parallel/mesh.py-75-    n_devices: int
packages/core/legoesm/parallel/mesh.py-76-    backend: str
packages/core/legoesm/parallel/mesh.py-77-    is_distributed: bool
packages/core/legoesm/parallel/mesh.py-78-    tiling: tuple[int, int] = (1, 1)
packages/core/legoesm/parallel/mesh.py-79-    grid_type: str = "cubed_sphere"
packages/core/legoesm/parallel/mesh.py:80:    voronoi_dims: tuple[int, int, int] | None = None
packages/core/legoesm/parallel/mesh.py-81-
packages/core/legoesm/parallel/mesh.py-82-
packages/core/legoesm/parallel/mesh.py-83-# Singleton — set once at startup, queried by the rest of the code.
packages/core/legoesm/parallel/mesh.py-84-_active_config: DeviceConfig | None = None
packages/core/legoesm/parallel/mesh.py-85-
--
packages/core/legoesm/parallel/mesh.py-643-
packages/core/legoesm/parallel/mesh.py-644-# ==============================================================================
packages/core/legoesm/parallel/mesh.py-645-# Voronoi (icosahedral) mesh parallelism
packages/core/legoesm/parallel/mesh.py-646-# ==============================================================================
packages/core/legoesm/parallel/mesh.py-647-
packages/core/legoesm/parallel/mesh.py:648:def create_voronoi_device_mesh(
packages/core/legoesm/parallel/mesh.py-649-    nCells: int,
packages/core/legoesm/parallel/mesh.py-650-    nEdges: int,
packages/core/legoesm/parallel/mesh.py-651-    nVertices: int,
packages/core/legoesm/parallel/mesh.py-652-    n_devices: int | str = "auto",
packages/core/legoesm/parallel/mesh.py-653-    backend: str | None = None,
packages/core/legoesm/parallel/mesh.py-654-    devices: Sequence | None = None,
packages/core/legoesm/parallel/mesh.py-655-) -> DeviceConfig:
packages/core/legoesm/parallel/mesh.py-656-    """Create a JAX device mesh for Voronoi (MPAS) mesh parallelism.
packages/core/legoesm/parallel/mesh.py-657-
packages/core/legoesm/parallel/mesh.py-658-    Shards cell- and edge-centered arrays along axis 0 across devices.
packages/core/legoesm/parallel/mesh.py:659:    For best performance, reorder the mesh with
packages/core/legoesm/parallel/mesh.py:660:    :func:`legoesm.parallel.voronoi_partition.reorder_voronoi_for_sharding`
packages/core/legoesm/parallel/mesh.py-661-    before sharding so that each device gets a spatially contiguous cell
packages/core/legoesm/parallel/mesh.py-662-    cluster.
packages/core/legoesm/parallel/mesh.py-663-
packages/core/legoesm/parallel/mesh.py-664-    Parameters
packages/core/legoesm/parallel/mesh.py-665-    ----------
--
packages/core/legoesm/parallel/mesh.py-677-        Optional explicit device list.
packages/core/legoesm/parallel/mesh.py-678-
packages/core/legoesm/parallel/mesh.py-679-    Returns
packages/core/legoesm/parallel/mesh.py-680-    -------
packages/core/legoesm/parallel/mesh.py-681-    DeviceConfig
packages/core/legoesm/parallel/mesh.py:682:        Config with ``grid_type="voronoi"`` and ``voronoi_dims`` set.
packages/core/legoesm/parallel/mesh.py-683-    """
packages/core/legoesm/parallel/mesh.py-684-    global _active_config
packages/core/legoesm/parallel/mesh.py-685-
packages/core/legoesm/parallel/mesh.py-686-    if devices is not None:
packages/core/legoesm/parallel/mesh.py-687-        devices = list(devices)
--
packages/core/legoesm/parallel/mesh.py-710-            n_devices=1,
packages/core/legoesm/parallel/mesh.py-711-            backend=backend_name,
packages/core/legoesm/parallel/mesh.py-712-            is_distributed=False,
packages/core/legoesm/parallel/mesh.py-713-            tiling=(1, 1),
packages/core/legoesm/parallel/mesh.py-714-            grid_type="voronoi",
packages/core/legoesm/parallel/mesh.py:715:            voronoi_dims=(nCells, nEdges, nVertices),
packages/core/legoesm/parallel/mesh.py-716-        )
packages/core/legoesm/parallel/mesh.py-717-        _active_config = config
packages/core/legoesm/parallel/mesh.py-718-        return config
packages/core/legoesm/parallel/mesh.py-719-
packages/core/legoesm/parallel/mesh.py-720-    selected = devices[:n_dev]
--
packages/core/legoesm/parallel/mesh.py-729-        n_devices=n_dev,
packages/core/legoesm/parallel/mesh.py-730-        backend=backend_name,
packages/core/legoesm/parallel/mesh.py-731-        is_distributed=False,
packages/core/legoesm/parallel/mesh.py-732-        tiling=(1, 1),
packages/core/legoesm/parallel/mesh.py-733-        grid_type="voronoi",
packages/core/legoesm/parallel/mesh.py:734:        voronoi_dims=(nCells, nEdges, nVertices),
packages/core/legoesm/parallel/mesh.py-735-    )
packages/core/legoesm/parallel/mesh.py-736-    _active_config = config
packages/core/legoesm/parallel/mesh.py-737-    logger.info(
packages/core/legoesm/parallel/mesh.py-738-        "legoESM: %d-device voronoi mesh on %s "
packages/core/legoesm/parallel/mesh.py:739:        "(nCells=%d, nEdges=%d, nVertices=%d)",
packages/core/legoesm/parallel/mesh.py:740:        n_dev, backend_name, nCells, nEdges, nVertices,
packages/core/legoesm/parallel/mesh.py-741-    )
packages/core/legoesm/parallel/mesh.py-742-    return config
packages/core/legoesm/parallel/mesh.py-743-
packages/core/legoesm/parallel/mesh.py-744-
packages/core/legoesm/parallel/mesh.py-745-# ==============================================================================
--
packages/core/legoesm/parallel/mesh.py-864-            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-865-
packages/core/legoesm/parallel/mesh.py-866-        elif config.grid_type == "voronoi":
packages/core/legoesm/parallel/mesh.py-867-            # Voronoi state arrays: shard cell- and edge-centered arrays
packages/core/legoesm/parallel/mesh.py-868-            # along axis 0.  Vertex arrays and small arrays are replicated.
packages/core/legoesm/parallel/mesh.py:869:            if config.voronoi_dims is not None and leaf.ndim >= 1:
packages/core/legoesm/parallel/mesh.py:870:                nCells, nEdges, _nVerts = config.voronoi_dims
packages/core/legoesm/parallel/mesh.py:871:                if leaf.shape[0] in (nCells, nEdges):
packages/core/legoesm/parallel/mesh.py-872-                    return multiprocess_safe_device_put(leaf, config.face_sharding)
packages/core/legoesm/parallel/mesh.py-873-            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-874-
packages/core/legoesm/parallel/mesh.py-875-        # Default: replicate
packages/core/legoesm/parallel/mesh.py-876-        return multiprocess_safe_device_put(leaf, config.replicated_sharding)
--
packages/core/legoesm/parallel/halo_exchange_voronoi.py-272-# rely on, which likewise reuse a fixed tag across every exchange call): two
packages/core/legoesm/parallel/halo_exchange_voronoi.py-273-# messages that share (comm, source, dest, tag) -- e.g. the production MPAS
packages/core/legoesm/parallel/halo_exchange_voronoi.py-274-# step's two cell exchanges (T+p_s, then tracers) between one pair -- are
packages/core/legoesm/parallel/halo_exchange_voronoi.py-275-# disambiguated NOT by the tag but by ORDER.  Every rank runs the same SPMD
packages/core/legoesm/parallel/halo_exchange_voronoi.py-276-# program order; mpi4jax's ordered effect keeps the sendrecvs from being
packages/core/legoesm/parallel/halo_exchange_voronoi.py:277:# reordered/parallelised; MPI's non-overtaking guarantee then pairs them 1:1.
packages/core/legoesm/parallel/halo_exchange_voronoi.py-278-# A NEW halo path that issues same-(source,dest,tag) messages MUST preserve that
packages/core/legoesm/parallel/halo_exchange_voronoi.py-279-# program-order property (regression: test_voronoi_mpi
packages/core/legoesm/parallel/halo_exchange_voronoi.py-280-# ::TestHaloExchange::test_repeated_same_entity_exchange_no_cross_match).
packages/core/legoesm/parallel/halo_exchange_voronoi.py-281-_BATCH_TAG_BASE = 8  # > the entity tags {0, 1, 2}; leaves headroom
packages/core/legoesm/parallel/halo_exchange_voronoi.py-282-
--
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1930-# corner-staggered D-grid winds carry a duplicated shared tile face that stays
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1931-# self-consistent across stages because adjacent tiles compute BIT-IDENTICAL
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1932-# shared-face du_d/dv_d (proven by the capstone gate's du/dv corner-overlap at
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1933-# 1e-10 — both tiles match the global there, hence each other).  Base cut: no
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1934-# post-step (sponge / damp_v / fix_ps_mass = increment-4); dphis=0 (phis const).
packages/core/legoesm/parallel/tiled_production_cdgrid.py:1935:# Bit-identical (to RK3-accumulated reorder) to ssp_rk3_step(state, the base-cut
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1936-# fv3_hydrostatic_tendencies, dt) — NOT the full _step_fv3 (which also does the
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1937-# post-step _sync_dgrid_boundary).  Future-HW (np>6 anti-scales on Ginsburg);
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1938-# gated by bit-identity, not wall-clock.
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1939-# ===========================================================================
packages/core/legoesm/parallel/tiled_production_cdgrid.py-1940-
--
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2558-# The FIRST global reduction in the cube tiled stages (all prior stages are
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2559-# halo-only) — the primitive the full tiled STEP needs for its mass-fixer and
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2560-# for the capstone's _apply_zero_mean_per_stage=True config path (the production
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2561-# default fix_mass=True path uses raw dp_s/dt, so the capstone's base cut omits
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2562-# it; this stage provides the alternative).  Bit-identical to the global
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2563:# core.conservation.zero_mean_tendency up to psum reduction-ORDER reordering;
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2564-# the conserved property sum(out*area)==0 holds to machine precision (the point).
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2565-# ---------------------------------------------------------------------------
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2566-
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2567-def make_tiled_zero_mean_tendency_stage_2d(mesh, grid, n: int, kt: int):
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2568-    """Tiled area-weighted ``zero_mean_tendency`` on a ``(6, kt, kt)`` mesh.
--
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2621-# Tiled GLOBAL reduction: dry-mass fixer fix_ps_mass via psum.  The cube tiled
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2622-# STEP's post-RK3 mass conservation (default use_conservation_fixer+fix_mass) —
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2623-# the second tiled global reduction (after zero_mean), and a TIER-0 truth
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2624-# (conservation) op.  Two area-weighted global sums (psum) of p_s_old/p_s_new ->
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2625-# a uniform additive p_s correction so dry mass is conserved.  Bit-identical to
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2626:# core.conservation.fix_ps_mass up to psum reduction-ORDER reordering; conserves
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2627-# sum(out*area)==sum(p_s_old*area) to machine precision (the point).
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2628-# ---------------------------------------------------------------------------
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2629-
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2630-def _tile_fix_ps_mass_delta(p_s_new_t, p_s_old_t, ar_t, total_area, acc):
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2631-    """DELTA-FIRST per-tile dry-mass fix (shared by the standalone fixer stage
--
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2635-    the original stage: two huge near-equal masses ~1e19 would catastrophically
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2636-    cancel in f32-storage / x64-off mode), ``psum``s the per-tile deltas over
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2637-    the ``(face, tile_i, tile_j)`` mesh axes, and applies the uniform additive
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2638-    correction.  ``total_area`` is the closed-over GLOBAL single-sum (matching
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2639-    the global op's ``_total_area`` — NOT a psummed local area, which would
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2640:    reorder the denominator off the global).  No cast-back (``fix_ps_mass``
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2641-    keeps the promoted dtype).  Must be called INSIDE a shard_map over the
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2642-    tiled mesh."""
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2643-    local_delta = jnp.sum(
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2644-        (p_s_old_t.astype(acc) - p_s_new_t.astype(acc)) * ar_t.astype(acc))
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2645-    g_delta = jax.lax.psum(
--
packages/core/legoesm/parallel/sharded_dynamics.py-1088-
packages/core/legoesm/parallel/sharded_dynamics.py-1089-# ======================================================================
packages/core/legoesm/parallel/sharded_dynamics.py-1090-# Voronoi (MPAS) multi-GPU sharded step
packages/core/legoesm/parallel/sharded_dynamics.py-1091-# ======================================================================
packages/core/legoesm/parallel/sharded_dynamics.py-1092-
packages/core/legoesm/parallel/sharded_dynamics.py:1093:def _pad_local_mesh_to(mesh, target_nCells, target_nEdges, target_nVertices):
packages/core/legoesm/parallel/sharded_dynamics.py-1094-    """Pad a local VoronoiMesh to target dimensions with inert ghost entities.
packages/core/legoesm/parallel/sharded_dynamics.py-1095-
packages/core/legoesm/parallel/sharded_dynamics.py-1096-    Ghost cells have ``areaCell=1``, zero signs/weights, and connectivity
packages/core/legoesm/parallel/sharded_dynamics.py-1097-    pointing to index 0.  Ghost edges have ``dvEdge=0`` (zero flux),
packages/core/legoesm/parallel/sharded_dynamics.py-1098-    ``dcEdge=1``, and ``cellsOnEdge=[0,0]``.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1245-#: a score computed at a different depth describes a different comm graph, and
packages/core/legoesm/parallel/sharded_dynamics.py-1246-#: two independently hardcoded 3s let production drift unnoticed.
packages/core/legoesm/parallel/sharded_dynamics.py-1247-SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py-1248-
packages/core/legoesm/parallel/sharded_dynamics.py-1249-
packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
packages/core/legoesm/parallel/sharded_dynamics.py:1251:                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py-1252-                       ppermute_cells_per_device_threshold=2_000):
packages/core/legoesm/parallel/sharded_dynamics.py-1253-    """How much halo communication one ownership choice costs, computed offline.
packages/core/legoesm/parallel/sharded_dynamics.py-1254-
packages/core/legoesm/parallel/sharded_dynamics.py-1255-    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
packages/core/legoesm/parallel/sharded_dynamics.py-1256-    ROUNDS one halo exchange needs -- the sequential collective launches that
--
packages/core/legoesm/parallel/sharded_dynamics.py-1280-      case there is no ppermute schedule and this number is counterfactual --
packages/core/legoesm/parallel/sharded_dynamics.py-1281-      see the returned ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py-1282-
packages/core/legoesm/parallel/sharded_dynamics.py-1283-    MESH STATE -- the one thing that silently invalidates the score
packages/core/legoesm/parallel/sharded_dynamics.py-1284-    --------------------------------------------------------------
packages/core/legoesm/parallel/sharded_dynamics.py:1285:    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
packages/core/legoesm/parallel/sharded_dynamics.py:1286:    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
packages/core/legoesm/parallel/sharded_dynamics.py:1287:    ONCE for a ``reorder_target`` device count and then runs at a possibly
packages/core/legoesm/parallel/sharded_dynamics.py-1288-    DIFFERENT device count.  So pass what you actually have:
packages/core/legoesm/parallel/sharded_dynamics.py-1289-
packages/core/legoesm/parallel/sharded_dynamics.py-1290-    * raw mesh, scoring a run at ``n_dev``: defaults are right.
packages/core/legoesm/parallel/sharded_dynamics.py:1291:    * raw mesh, but the run reorders for a different target: pass
packages/core/legoesm/parallel/sharded_dynamics.py:1292:      ``reorder_target=<that target>``; the split is built for the target and
packages/core/legoesm/parallel/sharded_dynamics.py-1293-      scored at ``n_dev``.
packages/core/legoesm/parallel/sharded_dynamics.py:1294:    * already-reordered mesh (what production holds): pass
packages/core/legoesm/parallel/sharded_dynamics.py:1295:      ``already_reordered=True``; ``method`` is then ignored and reported as
packages/core/legoesm/parallel/sharded_dynamics.py:1296:      ``"pre-reordered"``, because the ownership is already baked in.
packages/core/legoesm/parallel/sharded_dynamics.py-1297-
packages/core/legoesm/parallel/sharded_dynamics.py-1298-    Parameters
packages/core/legoesm/parallel/sharded_dynamics.py-1299-    ----------
packages/core/legoesm/parallel/sharded_dynamics.py-1300-    mesh : VoronoiMesh
packages/core/legoesm/parallel/sharded_dynamics.py-1301-    n_dev : int
packages/core/legoesm/parallel/sharded_dynamics.py-1302-        Device count the run uses.  Must be >= 1.
packages/core/legoesm/parallel/sharded_dynamics.py-1303-    method : str
packages/core/legoesm/parallel/sharded_dynamics.py:1304:        Ownership for the reorder; ignored when *already_reordered*.
packages/core/legoesm/parallel/sharded_dynamics.py:1305:    reorder_target : int | None
packages/core/legoesm/parallel/sharded_dynamics.py:1306:        Device count the reorder targets, when it differs from *n_dev*.
packages/core/legoesm/parallel/sharded_dynamics.py:1307:    already_reordered : bool
packages/core/legoesm/parallel/sharded_dynamics.py-1308-    halo_depth : int
packages/core/legoesm/parallel/sharded_dynamics.py-1309-        Must match production (3) or the graph is a different graph.
packages/core/legoesm/parallel/sharded_dynamics.py-1310-    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py-1311-        Mirror of the production auto-select threshold, only used to report
packages/core/legoesm/parallel/sharded_dynamics.py-1312-        ``production_strategy``.
--
packages/core/legoesm/parallel/sharded_dynamics.py-1323-    Reference census on the unrelaxed mesh, which any change here must still
packages/core/legoesm/parallel/sharded_dynamics.py-1324-    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
packages/core/legoesm/parallel/sharded_dynamics.py-1325-    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
packages/core/legoesm/parallel/sharded_dynamics.py-1326-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1327-    from legoesm.parallel.voronoi_partition import (
packages/core/legoesm/parallel/sharded_dynamics.py:1328:        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
packages/core/legoesm/parallel/sharded_dynamics.py-1329-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1330-
packages/core/legoesm/parallel/sharded_dynamics.py-1331-    if int(n_dev) != n_dev or int(n_dev) < 1:
packages/core/legoesm/parallel/sharded_dynamics.py-1332-        # int() would silently truncate 3.9 -> 3 and score the wrong split.
packages/core/legoesm/parallel/sharded_dynamics.py-1333-        raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py-1334-            f"spmd_schedule_cost: n_dev must be an integer >= 1, got {n_dev!r}")
packages/core/legoesm/parallel/sharded_dynamics.py-1335-    n_dev = int(n_dev)
packages/core/legoesm/parallel/sharded_dynamics.py-1336-
packages/core/legoesm/parallel/sharded_dynamics.py:1337:    if already_reordered:
packages/core/legoesm/parallel/sharded_dynamics.py:1338:        if reorder_target is not None:
packages/core/legoesm/parallel/sharded_dynamics.py-1339-            raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py:1340:                "spmd_schedule_cost: reorder_target is meaningless with "
packages/core/legoesm/parallel/sharded_dynamics.py:1341:                "already_reordered=True — the ownership is already baked into "
packages/core/legoesm/parallel/sharded_dynamics.py-1342-                "the mesh.")
packages/core/legoesm/parallel/sharded_dynamics.py:1343:        prepared, resolved = mesh, "pre-reordered"
packages/core/legoesm/parallel/sharded_dynamics.py-1344-    else:
packages/core/legoesm/parallel/sharded_dynamics.py:1345:        target = n_dev if reorder_target is None else int(reorder_target)
packages/core/legoesm/parallel/sharded_dynamics.py:1346:        prepared = reorder_voronoi_for_sharding(mesh, target, method=method)
packages/core/legoesm/parallel/sharded_dynamics.py-1347-        # Report the CONCRETE ownership: "auto" hides which partitioner ran.
packages/core/legoesm/parallel/sharded_dynamics.py:1348:        # Uses the SAME resolver the reorder used, so the label cannot drift
packages/core/legoesm/parallel/sharded_dynamics.py-1349-        # from the policy.
packages/core/legoesm/parallel/sharded_dynamics.py-1350-        resolved = resolve_sharding_partition_method(method)
packages/core/legoesm/parallel/sharded_dynamics.py-1351-
packages/core/legoesm/parallel/sharded_dynamics.py-1352-    # The builder assigns residual entities to the LAST owner but excludes them
packages/core/legoesm/parallel/sharded_dynamics.py-1353-    # from every owned contiguous block, so schedule send indices can exceed a
packages/core/legoesm/parallel/sharded_dynamics.py-1354-    # device's shard length -- a number that looks fine and is not.  Reachable
packages/core/legoesm/parallel/sharded_dynamics.py:1355:    # via reorder_target: a mesh padded for 3 devices is not divisible by 4.
packages/core/legoesm/parallel/sharded_dynamics.py-1356-    # The scaling bench rejects that pairing; so does this.
packages/core/legoesm/parallel/sharded_dynamics.py:1357:    n_cells, n_edges = int(prepared.nCells), int(prepared.nEdges)
packages/core/legoesm/parallel/sharded_dynamics.py-1358-    if n_cells % n_dev or n_edges % n_dev:
packages/core/legoesm/parallel/sharded_dynamics.py-1359-        raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py-1360-            f"spmd_schedule_cost: prepared mesh has nCells={n_cells}, "
packages/core/legoesm/parallel/sharded_dynamics.py-1361-            f"nEdges={n_edges}, neither divisible by n_dev={n_dev}. The mesh "
packages/core/legoesm/parallel/sharded_dynamics.py:1362:            f"is padded for its reorder target"
packages/core/legoesm/parallel/sharded_dynamics.py:1363:            f"{'' if already_reordered else f' ({target})'}, so scoring it at "
packages/core/legoesm/parallel/sharded_dynamics.py-1364-            f"a device count that does not divide it silently mis-slices the "
packages/core/legoesm/parallel/sharded_dynamics.py-1365-            f"owned blocks. Score at a device count that divides the prepared "
packages/core/legoesm/parallel/sharded_dynamics.py-1366-            f"mesh.")
packages/core/legoesm/parallel/sharded_dynamics.py-1367-    (
packages/core/legoesm/parallel/sharded_dynamics.py-1368-        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
--
packages/core/legoesm/parallel/sharded_dynamics.py-1374-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1375-    return {
packages/core/legoesm/parallel/sharded_dynamics.py-1376-        "method": method,
packages/core/legoesm/parallel/sharded_dynamics.py-1377-        "resolved_method": resolved,
packages/core/legoesm/parallel/sharded_dynamics.py-1378-        "n_dev": n_dev,
packages/core/legoesm/parallel/sharded_dynamics.py:1379:        # Unknown for a pre-reordered mesh: the ownership is baked in and the
packages/core/legoesm/parallel/sharded_dynamics.py-1380-        # target that produced it is not recoverable from the mesh. Reporting
packages/core/legoesm/parallel/sharded_dynamics.py-1381-        # n_dev there would assert something we did not verify.
packages/core/legoesm/parallel/sharded_dynamics.py:1382:        "reorder_target": (None if already_reordered else
packages/core/legoesm/parallel/sharded_dynamics.py:1383:                           (n_dev if reorder_target is None
packages/core/legoesm/parallel/sharded_dynamics.py:1384:                            else int(reorder_target))),
packages/core/legoesm/parallel/sharded_dynamics.py:1385:        "already_reordered": bool(already_reordered),
packages/core/legoesm/parallel/sharded_dynamics.py-1386-        "halo_depth": halo_depth,
packages/core/legoesm/parallel/sharded_dynamics.py-1387-        "n_rounds": int(sched["n_rounds"]),
packages/core/legoesm/parallel/sharded_dynamics.py-1388-        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
packages/core/legoesm/parallel/sharded_dynamics.py-1389-        "max_degree": int(sched.get("max_degree", -1)),
packages/core/legoesm/parallel/sharded_dynamics.py-1390-        "coloring_method": sched["coloring_method"],
--
packages/core/legoesm/parallel/sharded_dynamics.py-1401-
packages/core/legoesm/parallel/sharded_dynamics.py-1402-
packages/core/legoesm/parallel/sharded_dynamics.py-1403-def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
packages/core/legoesm/parallel/sharded_dynamics.py-1404-    """Pre-compute per-device local meshes and gather/scatter indices.
packages/core/legoesm/parallel/sharded_dynamics.py-1405-
packages/core/legoesm/parallel/sharded_dynamics.py:1406:    After ``reorder_voronoi_for_sharding`` the global mesh has *both*
packages/core/legoesm/parallel/sharded_dynamics.py-1407-    cells and edges ordered by contiguous device blocks.  We compute
packages/core/legoesm/parallel/sharded_dynamics.py-1408-    partitions whose owned-entity boundaries exactly match the shard
packages/core/legoesm/parallel/sharded_dynamics.py:1409:    boundaries (``cells_per = nCells // n_dev``, ``edges_per = nEdges //
packages/core/legoesm/parallel/sharded_dynamics.py-1410-    n_dev``), then build local meshes with remapped connectivity.
packages/core/legoesm/parallel/sharded_dynamics.py-1411-
packages/core/legoesm/parallel/sharded_dynamics.py-1412-    Using ``partition_voronoi_mesh`` directly is unsuitable because it
packages/core/legoesm/parallel/sharded_dynamics.py-1413-    derives edge ownership from cell ownership, producing an uneven edge
packages/core/legoesm/parallel/sharded_dynamics.py-1414-    split that mismatches the even shard split.  Instead we construct the
--
packages/core/legoesm/parallel/sharded_dynamics.py-1557-            missing = np.setdiff1d(owned_edges, local_edges_arr)
packages/core/legoesm/parallel/sharded_dynamics.py-1558-            raise RuntimeError(
packages/core/legoesm/parallel/sharded_dynamics.py-1559-                f"Voronoi partition rank={rank}: {len(missing)} owned "
packages/core/legoesm/parallel/sharded_dynamics.py-1560-                f"edges have a neighbour cell outside the halo-depth="
packages/core/legoesm/parallel/sharded_dynamics.py-1561-                f"{halo_depth} cell halo.  Increase halo_depth or check "
packages/core/legoesm/parallel/sharded_dynamics.py:1562:                f"the mesh ordering produced by reorder_voronoi_for_sharding."
packages/core/legoesm/parallel/sharded_dynamics.py-1563-            )
packages/core/legoesm/parallel/sharded_dynamics.py-1564-        halo_edges = np.setdiff1d(
packages/core/legoesm/parallel/sharded_dynamics.py-1565-            local_edges_arr, owned_edges, assume_unique=True,
packages/core/legoesm/parallel/sharded_dynamics.py-1566-        )
packages/core/legoesm/parallel/sharded_dynamics.py-1567-        local_edges = np.concatenate([owned_edges, halo_edges])
--
packages/core/legoesm/parallel/sharded_dynamics.py-1714-def _multi_ordering_edge_coloring(comm_pairs):
packages/core/legoesm/parallel/sharded_dynamics.py-1715-    """Proper edge coloring via multi-start first-fit; returns the coloring
packages/core/legoesm/parallel/sharded_dynamics.py-1716-    using the FEWEST colors (= ppermute rounds) across several deterministic
packages/core/legoesm/parallel/sharded_dynamics.py-1717-    visitation orders.
packages/core/legoesm/parallel/sharded_dynamics.py-1718-
packages/core/legoesm/parallel/sharded_dynamics.py:1719:    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
packages/core/legoesm/parallel/sharded_dynamics.py-1720-    the sorted order can overshoot the chromatic index by up to 3 rounds at
packages/core/legoesm/parallel/sharded_dynamics.py-1721-    16 devices, while a degree-descending or shuffled order reaches the
packages/core/legoesm/parallel/sharded_dynamics.py-1722-    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
packages/core/legoesm/parallel/sharded_dynamics.py-1723-    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
packages/core/legoesm/parallel/sharded_dynamics.py-1724-    coloring by construction, so taking the min can NEVER produce an
--
packages/core/legoesm/parallel/sharded_dynamics.py-2227-    from legoesm.timestepping.integration import (
packages/core/legoesm/parallel/sharded_dynamics.py-2228-        refuse_unthreaded_stateful_physics,
packages/core/legoesm/parallel/sharded_dynamics.py-2229-    )
packages/core/legoesm/parallel/sharded_dynamics.py-2230-
packages/core/legoesm/parallel/sharded_dynamics.py-2231-    n_dev = dev_config.n_devices
packages/core/legoesm/parallel/sharded_dynamics.py:2232:    voronoi_dims = dev_config.voronoi_dims
packages/core/legoesm/parallel/sharded_dynamics.py:2233:    if voronoi_dims is None:
packages/core/legoesm/parallel/sharded_dynamics.py:2234:        raise ValueError("dev_config.voronoi_dims must be set for Voronoi grids")
packages/core/legoesm/parallel/sharded_dynamics.py:2235:    nCells, nEdges, _nVerts = voronoi_dims
packages/core/legoesm/parallel/sharded_dynamics.py-2236-    jax_mesh = dev_config.mesh
packages/core/legoesm/parallel/sharded_dynamics.py-2237-
packages/core/legoesm/parallel/sharded_dynamics.py-2238-    cells_per = nCells // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-2239-    edges_per = nEdges // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-2240-
--
packages/core/legoesm/parallel/sharded_dynamics.py-2278-    # ------------------------------------------------------------------
packages/core/legoesm/parallel/sharded_dynamics.py-2279-    # Setup: build per-device local meshes and gather indices
packages/core/legoesm/parallel/sharded_dynamics.py-2280-    # ------------------------------------------------------------------
packages/core/legoesm/parallel/sharded_dynamics.py-2281-    logger.info(
packages/core/legoesm/parallel/sharded_dynamics.py-2282-        "Building halo-partitioned infrastructure for %d device(s) "
packages/core/legoesm/parallel/sharded_dynamics.py:2283:        "(nCells=%d, nEdges=%d, halo_depth=3, strategy=%s) ...",
packages/core/legoesm/parallel/sharded_dynamics.py:2284:        n_dev, nCells, nEdges, halo_strategy,
packages/core/legoesm/parallel/sharded_dynamics.py-2285-    )
packages/core/legoesm/parallel/sharded_dynamics.py-2286-    t0 = time.time()
packages/core/legoesm/parallel/sharded_dynamics.py-2287-    (
packages/core/legoesm/parallel/sharded_dynamics.py-2288-        stacked_meshes,   # VoronoiMesh pytree with (n_dev, max_l*) leaves
packages/core/legoesm/parallel/sharded_dynamics.py-2289-        gather_cells,     # (n_dev, max_lc)
--
packages/core/legoesm/parallel/sharded_dynamics.py-2358-        total_pp_bytes = sum(
packages/core/legoesm/parallel/sharded_dynamics.py-2359-            hc * (nlev + 2) + he * nlev
packages/core/legoesm/parallel/sharded_dynamics.py-2360-            for hc, he in zip(pp_sched['halo_cells_per_round'],
packages/core/legoesm/parallel/sharded_dynamics.py-2361-                              pp_sched['halo_edges_per_round'])
packages/core/legoesm/parallel/sharded_dynamics.py-2362-        ) * 4  # float32
packages/core/legoesm/parallel/sharded_dynamics.py:2363:        ag_bytes = (nCells * (nlev + 2) + nEdges * nlev) * 4
packages/core/legoesm/parallel/sharded_dynamics.py-2364-        logger.info(
packages/core/legoesm/parallel/sharded_dynamics.py-2365-            "  ppermute schedule: %d rounds, max halo cells/edges per round: %s / %s",
packages/core/legoesm/parallel/sharded_dynamics.py-2366-            n_rounds,
packages/core/legoesm/parallel/sharded_dynamics.py-2367-            pp_sched['halo_cells_per_round'],
packages/core/legoesm/parallel/sharded_dynamics.py-2368-            pp_sched['halo_edges_per_round'],
--
packages/core/legoesm/parallel/device_config.py-582-    - Level-parallel sharding when nlev > device_count.
packages/core/legoesm/parallel/device_config.py-583-    - Falls back to single-device when device_count == 1.
packages/core/legoesm/parallel/device_config.py-584-
packages/core/legoesm/parallel/device_config.py-585-    **Voronoi (icosahedral):**
packages/core/legoesm/parallel/device_config.py-586-    - Cell/edge dimension sharding across all devices.
packages/core/legoesm/parallel/device_config.py:587:    - Reorder mesh with ``reorder_voronoi_for_sharding`` for best locality.
packages/core/legoesm/parallel/device_config.py-588-    """
scripts/bench/bench_spectral_transform_micro.py-173-    else:
scripts/bench/bench_spectral_transform_micro.py-174-        os.environ["LEGOESM_SH_GEMM"] = _prev_gemm_env
scripts/bench/bench_spectral_transform_micro.py-175-
scripts/bench/bench_spectral_transform_micro.py-176-    payload = {
scripts/bench/bench_spectral_transform_micro.py-177-        "rows": rows,
scripts/bench/bench_spectral_transform_micro.py:178:        "metadata": annotate_incomplete(scaling_metadata(
scripts/bench/bench_spectral_transform_micro.py-179-            grid="spectral",
scripts/bench/bench_spectral_transform_micro.py-180-            component="atmosphere",
scripts/bench/bench_spectral_transform_micro.py-181-            resolution=",".join(f"T{t}" for t in truncs),
scripts/bench/bench_spectral_transform_micro.py-182-            n_levels=args.nlev,
scripts/bench/bench_spectral_transform_micro.py-183-            precision="float64",
--
scripts/bench/bench_mpas_spmd_scaling.py-537-        backend=jax.default_backend(),
scripts/bench/bench_mpas_spmd_scaling.py-538-        **tidy_throughput_fields(
scripts/bench/bench_mpas_spmd_scaling.py-539-            dt_seconds=dt, time_per_step_ms=med,
scripts/bench/bench_mpas_spmd_scaling.py-540-            total_cells=int(mesh.nCells) * args.nlev),
scripts/bench/bench_mpas_spmd_scaling.py-541-    )
scripts/bench/bench_mpas_spmd_scaling.py:542:    rec["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/bench_mpas_spmd_scaling.py-543-        grid="icosahedral",
scripts/bench/bench_mpas_spmd_scaling.py-544-        component="atmosphere",
scripts/bench/bench_mpas_spmd_scaling.py-545-        resolution=f"L{args.subdivision}",
scripts/bench/bench_mpas_spmd_scaling.py-546-        n_levels=args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py-547-        precision="float64" if jax.config.jax_enable_x64 else "float32",
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-812-        **comm_rec,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-813-        **bound_rec,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-814-    )
scripts/bench/bench_ocean_latlon_spmd_scaling.py-815-    from legoesm.parallel.early_init import nccl_transport_report
scripts/bench/bench_ocean_latlon_spmd_scaling.py-816-    _nccl_report = nccl_transport_report()
scripts/bench/bench_ocean_latlon_spmd_scaling.py:817:    rec["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-818-        grid="tripole" if args.tripole else "latlon",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-819-        component="ocean",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-820-        resolution=f"{n_lat}x{args.n_lon}",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-821-        n_levels=args.nlev,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-822-        precision="float64" if jax.config.jax_enable_x64 else "float32",
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-469-                   "t_bound_ms", "measured_over_bound"):
scripts/bench/bench_atm_latlon_spmd_scaling.py-470-            if _k in rec:
scripts/bench/bench_atm_latlon_spmd_scaling.py-471-                rec[_k] = None
scripts/bench/bench_atm_latlon_spmd_scaling.py-472-    from legoesm.parallel.early_init import nccl_transport_report
scripts/bench/bench_atm_latlon_spmd_scaling.py-473-    _nccl_report = nccl_transport_report()
scripts/bench/bench_atm_latlon_spmd_scaling.py:474:    rec["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/bench_atm_latlon_spmd_scaling.py-475-        grid="latlon",
scripts/bench/bench_atm_latlon_spmd_scaling.py-476-        component="atmosphere",
scripts/bench/bench_atm_latlon_spmd_scaling.py-477-        resolution=f"{n_lat}x{args.n_lon}",
scripts/bench/bench_atm_latlon_spmd_scaling.py-478-        n_levels=args.nlev,
scripts/bench/bench_atm_latlon_spmd_scaling.py-479-        precision="float64" if jax.config.jax_enable_x64 else "float32",
--
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
--
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
--
scripts/bench/run_levante_gpu_scaling.py-2388-                "cells_per_device": r.cells_per_gpu,
scripts/bench/run_levante_gpu_scaling.py-2389-            },
scripts/bench/run_levante_gpu_scaling.py-2390-        )
scripts/bench/run_levante_gpu_scaling.py-2391-        if metadata_overrides:
scripts/bench/run_levante_gpu_scaling.py-2392-            md_kwargs.update(metadata_overrides)
scripts/bench/run_levante_gpu_scaling.py:2393:        d["metadata"] = annotate_incomplete(scaling_metadata(**md_kwargs))
scripts/bench/run_levante_gpu_scaling.py-2394-        return d
scripts/bench/run_levante_gpu_scaling.py-2395-
scripts/bench/run_levante_gpu_scaling.py-2396-    payload = {
scripts/bench/run_levante_gpu_scaling.py-2397-        "mode": report.mode,
scripts/bench/run_levante_gpu_scaling.py-2398-        "precisions": report.precisions,
--
scripts/bench/metadata.py-17-
scripts/bench/metadata.py-18-Consumers: ``run_levante_gpu_scaling.py``, ``run_cpu_mpi_scaling.py``,
scripts/bench/metadata.py-19-``bench_atm_latlon_spmd_scaling.py``, ``bench_ocean_latlon_spmd_scaling.py``,
scripts/bench/metadata.py-20-``bench_mpas_spmd_scaling.py``, ``bench_ocean_mpi_scaling.py``,
scripts/bench/metadata.py-21-``bench_ocean_gpu_scaling.py`` (and any future bench driver) merge
scripts/bench/metadata.py:22:``scaling_metadata(...)`` under the ``"metadata"`` key of their JSON payload.
scripts/bench/metadata.py-23-Aggregators read ``payload["metadata"]``.
scripts/bench/metadata.py-24-"""
scripts/bench/metadata.py-25-from __future__ import annotations
scripts/bench/metadata.py-26-
scripts/bench/metadata.py-27-import os
--
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
--
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
--
scripts/bench/metadata.py-544-            "cannot be compared must not be written (roadmap benchmark hygiene)."
scripts/bench/metadata.py-545-        )
scripts/bench/metadata.py-546-    return problems
scripts/bench/metadata.py-547-
scripts/bench/metadata.py-548-
scripts/bench/metadata.py:549:def annotate_incomplete(md: dict[str, Any], *, warn: bool = True) -> dict[str, Any]:
scripts/bench/metadata.py-550-    """Flag (do NOT discard) an incomplete record at write time.
scripts/bench/metadata.py-551-
scripts/bench/metadata.py-552-    A benchmark record is built AFTER an expensive run has already completed,
scripts/bench/metadata.py-553-    so a hard raise here would throw away real data.  Instead: validate
scripts/bench/metadata.py-554-    non-strictly, and if the record is not self-describing, embed the problem
scripts/bench/metadata.py-555-    list under ``md["_incomplete"]`` and emit a ``RuntimeWarning`` so the row
scripts/bench/metadata.py-556-    is LOUDLY flagged and a downstream aggregator can skip/annotate it.
scripts/bench/metadata.py-557-    Returns ``md`` (mutated) for chaining.  Use this at JSON-write time; use
scripts/bench/metadata.py-558-    :func:`validate_scaling_metadata` (strict) where aborting is acceptable.
scripts/bench/metadata.py-559-    """
scripts/bench/metadata.py:560:    problems = validate_scaling_metadata(md, strict=False)
scripts/bench/metadata.py-561-    if problems:
scripts/bench/metadata.py-562-        md["_incomplete"] = problems
scripts/bench/metadata.py-563-        if warn:
scripts/bench/metadata.py-564-            import warnings
scripts/bench/metadata.py-565-
--
scripts/bench/bench_voronoi_partition_methods.py-330-        "auto_resolves_to": resolve_partition_method("auto"),
scripts/bench/bench_voronoi_partition_methods.py-331-        "step_time_pointer": (
scripts/bench/bench_voronoi_partition_methods.py-332-            "step-time per method: bench_ocean_mpas_scaling.py / "
scripts/bench/bench_voronoi_partition_methods.py-333-            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
scripts/bench/bench_voronoi_partition_methods.py-334-            "per launch, same case otherwise)"),
scripts/bench/bench_voronoi_partition_methods.py:335:        "metadata": annotate_incomplete(scaling_metadata(
scripts/bench/bench_voronoi_partition_methods.py-336-            grid="voronoi",
scripts/bench/bench_voronoi_partition_methods.py-337-            component="partitioning",
scripts/bench/bench_voronoi_partition_methods.py-338-            resolution=f"L{args.subdivision}",
scripts/bench/bench_voronoi_partition_methods.py-339-            n_levels=0,
scripts/bench/bench_voronoi_partition_methods.py-340-            precision="n/a",
--
scripts/bench/run_cpu_mpi_scaling.py-1439-    # comparable and a host-staged or f32 run is falsifiable from the record.
scripts/bench/run_cpu_mpi_scaling.py-1440-    # cs-spmd: result.n_ranks is the DEVICE count (rewritten upstream); leave
scripts/bench/run_cpu_mpi_scaling.py-1441-    # metadata n_ranks to auto process-count.  Non-cs-spmd (mpi4jax): jax is
scripts/bench/run_cpu_mpi_scaling.py-1442-    # unaware of the MPI world, so the real MPI rank count must be passed.
scripts/bench/run_cpu_mpi_scaling.py-1443-    _md_n_ranks = None if cs_spmd else result.n_ranks
scripts/bench/run_cpu_mpi_scaling.py:1444:    payload["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/run_cpu_mpi_scaling.py-1445-        grid=result.grid_type,
scripts/bench/run_cpu_mpi_scaling.py-1446-        component="atmosphere",
scripts/bench/run_cpu_mpi_scaling.py-1447-        resolution=result.resolution,
scripts/bench/run_cpu_mpi_scaling.py-1448-        n_levels=result.n_levels,
scripts/bench/run_cpu_mpi_scaling.py-1449-        precision=result.precision,

exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '90,170p'; nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '175,335p'; nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '480,570p'; nl -ba scripts/bench/metadata.py | sed -n '413,526p'; nl -ba scripts/bench/metadata.py | sed -n '527,574p'; rg -n -C 4 \"mpas.*spmd|bench_mpas_spmd_scaling|--reorder-for|partition-method\" scripts/cluster/scaling_levante -g '*.sbatch' -g '*.slurm' -g '*.sh' | head -n 500" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 526ms:
    90	}
    91	MPAS_PARITY_MAX_STEPS = 8
    92	
    93	# Conservation gate default: with fix_mass=True the step restores the global
    94	# dry mass to the pre-step value each step, so the drift over a smoke window
    95	# is the allreduce rounding floor, not scheme drift.
    96	MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
    97	
    98	
    99	def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
   100	                          moist=False, lloyd_iterations=50):
   101	    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
   102	
   103	    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
   104	    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
   105	    device count of THIS run's mesh/model (the two differ for the
   106	    single-device reference leg of a ladder, via ``--reorder-for``).
   107	    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
   108	    wave) so the sharded step's packed tracer halo exchange + RK tracer
   109	    advection sit on the timed/gated path.
   110	    """
   111	    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
   112	        MPASPrimitiveEquationConfig,
   113	        MPASPrimitiveEquationModel,
   114	    )
   115	    from legoesm.grids.vertical import create_sigma_coordinate
   116	    from legoesm.grids.voronoi import create_voronoi_mesh
   117	    from legoesm.parallel.mesh import create_voronoi_device_mesh
   118	    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
   119	
   120	    mesh = create_voronoi_mesh(subdivision_level=subdivision,
   121	                               lloyd_iterations=lloyd_iterations)
   122	    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
   123	    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
   124	        # Padding only guarantees divisibility for reorder_target.
   125	        raise SystemExit(
   126	            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
   127	            f"divisible by --n-devices {run_nd}; use a ladder where every "
   128	            f"count divides --reorder-for ({reorder_target}).")
   129	    sigma = create_sigma_coordinate(nlev)
   130	    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
   131	    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
   132	    # energy-conserving PV flux, SSP-RK3, global mass fixer.
   133	    cfg = MPASPrimitiveEquationConfig(
   134	        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
   135	        pv_scheme="energy", time_integrator="ssp_rk3",
   136	    )
   137	    dev_config = create_voronoi_device_mesh(
   138	        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
   139	        n_devices=run_nd,
   140	    )
   141	    if dev_config.n_devices > 1:
   142	        from legoesm.parallel.mesh import replicate_pytree
   143	        mesh_model = replicate_pytree(mesh, dev_config)
   144	    else:
   145	        mesh_model = mesh
   146	    model = MPASPrimitiveEquationModel(mesh_model, sigma, cfg)
   147	    # #1100 MPAS twin: the timed path never global-builds the state.
   148	    # build_sharded_baroclinic_wave_state_mpas creates every leaf via
   149	    # jax.make_array_from_callback (only THIS process's shard rows are
   150	    # ever materialised; value-identical (few-ULP contract, measured
   151	    # exact on the pinned CPU stack) to global-build + shard_pytree —
   152	    # tests/parallel/test_mpas_partitionlocal_build.py).  The GLOBAL
   153	    # state is built lazily in main() only for the parity/conservation
   154	    # gates (small smoke scales).  The mesh itself is still global per
   155	    # process — its SFC-partition-local construction is the open
   156	    # remainder of #1100.
   157	    from tests.test_cases.baroclinic_wave import (
   158	        build_sharded_baroclinic_wave_state_mpas,
   159	    )
   160	    state_sharded = build_sharded_baroclinic_wave_state_mpas(
   161	        mesh, sigma, dev_config, perturbed=True, moist=moist)
   162	    return mesh, model, state_sharded, dev_config
   163	
   164	
   165	def _block(state):
   166	    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
   167	                           if leaf is not None])
   168	
   169	
   170	def _global_dry_mass(state, mesh):
   175	
   176	
   177	def main() -> int:
   178	    p = argparse.ArgumentParser()
   179	    p.add_argument("--subdivision", type=int, default=5,
   180	                   help="icosahedral subdivision level L "
   181	                        "(nCells = 10*4^L + 2 before ghost padding)")
   182	    p.add_argument("--nlev", type=int, default=8)
   183	    p.add_argument("--lloyd", type=int, default=50,
   184	                   help="Lloyd relaxation iterations for the mesh. 50 = "
   185	                        "production SCVT; 0 = labelled synthetic scaling "
   186	                        "mesh (scaling receipts only, never physics — "
   187	                        "must match the prewarmed cache key at subdiv>=9).")
   188	    p.add_argument("--n-devices", type=int, required=True)
   189	    p.add_argument("--reorder-for", type=int, default=None,
   190	                   help="partition/reorder the mesh for THIS device count "
   191	                        "(default: --n-devices). Pin it to the ladder's "
   192	                        "max so single-device reference runs time the "
   193	                        "identical reordered mesh.")
   194	    p.add_argument("--partition-method",
   195	                   choices=["auto", "geometric", "metis", "sfc"],
   196	                   default="auto")
   197	    p.add_argument("--physics", choices=["none", "held_suarez", "kessler"],
   198	                   default="none",
   199	                   help="Operator-split physics on the timed path. "
   200	                        "'kessler' also attaches the q_v/q_c/q_r moist-"
   201	                        "baroclinic-wave tracers (packed tracer halo "
   202	                        "exchange + RK tracer advection on the gated "
   203	                        "path) and extends the parity gate to the "
   204	                        "tracer fields.")
   205	    p.add_argument("--halo-strategy",
   206	                   choices=["auto", "ppermute", "allgather"],
   207	                   default="auto",
   208	                   help="Halo strategy for make_voronoi_sharded_step. "
   209	                        "'auto' picks allgather below the per-device "
   210	                        "cell threshold — force 'ppermute' to exercise "
   211	                        "the neighbor-round schedule on small gate "
   212	                        "meshes (the multicontroller selfspawn tests "
   213	                        "do).  Recorded in the JSONL row.")
   214	    p.add_argument("--steps", type=int, default=12)
   215	    p.add_argument("--warmup", type=int, default=2)
   216	    p.add_argument("--dt", type=float, default=None,
   217	                   help="timestep [s]; default auto: 600 * 4**(4-L) "
   218	                        "(CFL: dx halves per level), min 30 s.")
   219	    p.add_argument("--out", type=str,
   220	                   default="results/a1/mpas_spmd_scaling.jsonl")
   221	    p.add_argument(
   222	        "--parity-gate", action="store_true",
   223	        help="Correctness gate: compare the gathered sharded trajectory "
   224	             "against the single-device model.step trajectory on the SAME "
   225	             "reordered mesh (smoke windows only; the re-association floor "
   226	             "grows with steps).")
   227	    p.add_argument(
   228	        "--check-conservation", action="store_true",
   229	        help="Gate global dry-mass drift sum(p_s*areaCell) over the run "
   230	             "(pre-shard state vs gathered final state; exits nonzero on "
   231	             "breach).")
   232	    p.add_argument("--mass-rtol", type=float, default=None,
   233	                   help="Conservation tolerance (default: 1e-11 f64 / "
   234	                        "1e-5 f32 — fix_mass pins the mass each step).")
   235	    p.add_argument("--multicontroller", action="store_true",
   236	                   help="Route-B multi-controller: jax.distributed.initialize "
   237	                        "per process, ('device',) mesh over the GLOBAL device "
   238	                        "set (one process per GPU / per CPU-device group). NO "
   239	                        "mpi4jax. --n-devices must equal the global device "
   240	                        "count.")
   241	    p.add_argument("--coordinator", type=str, default=None,
   242	                   help="host:port for jax.distributed when auto-detection "
   243	                        "(SLURM) is unavailable; process count/id then come "
   244	                        "from OMPI_COMM_WORLD_SIZE/RANK.")
   245	    args = p.parse_args()
   246	
   247	    # Validate the timing window BEFORE any model/device work (codex, ocean
   248	    # twin): an empty steady slice would only fail after the expensive run.
   249	    if args.steps < 1:
   250	        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
   251	    if not (0 <= args.warmup < args.steps):
   252	        raise SystemExit(
   253	            f"--warmup must satisfy 0 <= warmup < steps "
   254	            f"(got warmup={args.warmup}, steps={args.steps})")
   255	    if args.parity_gate and args.steps > MPAS_PARITY_MAX_STEPS:
   256	        raise SystemExit(
   257	            f"--parity-gate is a smoke gate (re-association floor grows "
   258	            f"with steps); --steps {args.steps} > {MPAS_PARITY_MAX_STEPS} "
   259	            f"cap.")
   260	
   261	    if args.multicontroller:
   262	        # MUST run before any other JAX use (backend init). SLURM auto-detects;
   263	        # mpiexec needs the explicit coordinator + launcher env vars (OpenMPI
   264	        # OMPI_*, or Cray PALS PMI_* on Derecho).
   265	        if args.coordinator is not None:
   266	            n_procs = int(os.environ.get(
   267	                "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "0")))
   268	            proc_id = int(os.environ.get(
   269	                "OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "-1")))
   270	            if n_procs < 1 or proc_id < 0:
   271	                raise SystemExit(
   272	                    "--coordinator given but no launcher rank env found "
   273	                    "(OMPI_COMM_WORLD_SIZE/RANK or PMI_SIZE/PMI_RANK).")
   274	            jax.distributed.initialize(
   275	                coordinator_address=args.coordinator,
   276	                num_processes=n_procs, process_id=proc_id)
   277	        else:
   278	            # Environment-routed: SLURM/OMPI -> bare auto-detect; PALS/PMI
   279	            # (Derecho mpiexec) -> mpi4py bootstrap. Real init failures
   280	            # re-raise loudly.
   281	            from legoesm.parallel.early_init import (
   282	                init_jax_distributed_with_fallback,
   283	            )
   284	            init_jax_distributed_with_fallback()
   285	
   286	    from legoesm.parallel.sharded_dynamics import (
   287	        gather_voronoi_state_spmd,
   288	        make_voronoi_sharded_step,
   289	    )
   290	
   291	    nd = args.n_devices
   292	    avail = len(jax.devices())
   293	    if avail < nd:
   294	        raise SystemExit(f"need {nd} devices, have {avail} "
   295	                         f"(set --xla_force_host_platform_device_count)")
   296	    if args.multicontroller and nd != avail:
   297	        # A mesh over a strict subset would leave some processes' devices out
   298	        # of the program (non-addressable participation hazard). Route-B uses
   299	        # ALL global devices: one shard per device across every process.
   300	        raise SystemExit(
   301	            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
   302	            f"device count ({avail} across {jax.process_count()} processes).")
   303	
   304	    dt = args.dt
   305	    if dt is None:
   306	        dt = max(600.0 * 4.0 ** (4 - args.subdivision), 30.0)
   307	
   308	    reorder_for = args.reorder_for if args.reorder_for is not None else nd
   309	    if reorder_for < nd:
   310	        raise SystemExit(
   311	            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
   312	            f"the ghost padding only guarantees divisibility for the "
   313	            f"partition target.")
   314	    mesh, model, s0, dev_config = build_model_and_state(
   315	        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
   316	        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
   317	
   318	    if args.multicontroller:
   319	        # Every process computed the reorder independently — assert the
   320	        # partitions agree before any collective uses the halo schedule.
   321	        # The checksum covers the entity ORDER (coordinates) and the
   322	        # connectivity the ppermute schedule + TRiSK stencils read; a
   323	        # rank-divergent partition (e.g. one rank resolving
   324	        # --partition-method auto to METIS, another to RCB) cannot slip
   325	        # through on cell positions alone.
   326	        from jax.experimental import multihost_utils
   327	        crc = 0
   328	        for arr, dtype in (
   329	            (mesh.latCell, np.float64), (mesh.latEdge, np.float64),
   330	            (mesh.cellsOnEdge, np.int64), (mesh.edgesOnCell, np.int64),
   331	            (mesh.cellsOnCell, np.int64), (mesh.areaCell, np.float64),
   332	        ):
   333	            crc = zlib.crc32(np.ascontiguousarray(
   334	                np.asarray(arr, dtype=dtype)).tobytes(), crc)
   335	        crc = zlib.crc32(
   480	                got = np.asarray(got)
   481	                field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
   482	                ok &= field_ok
   483	                if rank0:
   484	                    mx = (float(np.max(np.abs(got - want)))
   485	                          if want.size else 0.0)
   486	                    print(f"    parity {name:>4s}: max|diff|={mx:.3e} "
   487	                          f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
   488	            if not ok:
   489	                if rank0:
   490	                    print("ERROR: SPMD parity gate MISMATCH vs the "
   491	                          "single-device reference.", flush=True)
   492	                return 5
   493	
   494	    steady = per_step_ms[args.warmup:]
   495	    med = float(np.median(steady))
   496	    rec = dict(
   497	        component="mpas_atm",
   498	        subdivision=args.subdivision, n_devices=nd,
   499	        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
   500	        partition_method=args.partition_method, physics=args.physics,
   501	        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
   502	        # a row without this field could pass as a production-SCVT receipt.
   503	        lloyd_iterations=args.lloyd,
   504	        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
   505	        # saying "auto" would not reveal whether ppermute or allgather
   506	        # was actually measured (codex M3c-2 MINOR).
   507	        halo_strategy_requested=args.halo_strategy,
   508	        halo_strategy_effective=getattr(
   509	            step, "_halo_strategy_effective", "serial"),
   510	        steps=args.steps, dt=dt,
   511	        platform=jax.default_backend(),
   512	        n_processes=jax.process_count(),
   513	        multicontroller=bool(args.multicontroller),
   514	        compile_ms=round(per_step_ms[0], 1),
   515	        steady_median_ms=round(med, 2),
   516	        steady_min_ms=round(float(np.min(steady)), 2),
   517	        per_step_ms=[round(x, 1) for x in per_step_ms],
   518	        cells=int(mesh.nCells) * args.nlev,
   519	        # ppermute round count/step (static compile property; #1113) — the
   520	        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
   521	        # belongs on every row like the cube benches.
   522	        hlo_collective_permutes=hlo_cp,
   523	        # full per-family census (permute + all-reduce + all-gather + ...) on
   524	        # the SAME compile: exposes any reduction the ico step introduces.
   525	        hlo_collectives=hlo_census,
   526	    )
   527	    # Flat aggregator-compatible identity + metric fields (see the latlon
   528	    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
   529	    # icosahedral convention so both lanes land on the same plot curves.
   530	    rec.update(
   531	        grid_type="icosahedral",
   532	        resolution=args.subdivision,
   533	        n_levels=args.nlev,
   534	        mode="strong",  # this bench fixes the mesh and sweeps devices
   535	        precision="float64" if jax.config.jax_enable_x64 else "float32",
   536	        physics_level=args.physics,
   537	        backend=jax.default_backend(),
   538	        **tidy_throughput_fields(
   539	            dt_seconds=dt, time_per_step_ms=med,
   540	            total_cells=int(mesh.nCells) * args.nlev),
   541	    )
   542	    rec["metadata"] = annotate_incomplete(scaling_metadata(
   543	        grid="icosahedral",
   544	        component="atmosphere",
   545	        resolution=f"L{args.subdivision}",
   546	        n_levels=args.nlev,
   547	        precision="float64" if jax.config.jax_enable_x64 else "float32",
   548	        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
   549	                else 0),
   550	        decomposition="cell_partition" if nd > 1 else "none",
   551	        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
   552	        # share lives in extra.cells_per_device — a single-process 4-device
   553	        # SPMD run has 1 rank owning ALL cells (codex finding 3).
   554	        cells_per_rank=int(mesh.nCells) * args.nlev
   555	        // max(jax.process_count(), 1),
   556	        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
   557	        extra={
   558	            "partition_method": args.partition_method,
   559	            "physics": args.physics,
   560	            "steps": args.steps,
   561	            "multicontroller": bool(args.multicontroller),
   562	            "cells_per_device": int(mesh.nCells) // nd * args.nlev,
   563	        },
   564	    ))
   565	    # Multi-controller: every process times the same program; process 0 owns
   566	    # the JSONL + stdout (others would duplicate/corrupt the append).
   567	    if jax.process_index() == 0:
   568	        _outdir = os.path.dirname(args.out)
   569	        if _outdir:  # a bare basename --out needs no mkdir
   570	            os.makedirs(_outdir, exist_ok=True)
   413	def scaling_metadata(
   414	    *,
   415	    grid: str,
   416	    component: str,
   417	    resolution: Any,
   418	    n_levels: int,
   419	    precision: str,
   420	    n_ranks: int | None = None,
   421	    n_gpus: int = 0,
   422	    devices_per_rank: int | None = None,
   423	    decomposition: str = "none",
   424	    solver_variant: str = "n/a",
   425	    solver_residual: float | None = None,
   426	    conservation_drift: float | None = None,
   427	    cells_per_rank: int | None = None,
   428	    scaling_kind: str | None = None,
   429	    transport: str | None = None,
   430	    partition_metrics: dict[str, Any] | None = None,
   431	    timestamp_utc: str | None = None,
   432	    extra: dict[str, Any] | None = None,
   433	) -> dict[str, Any]:
   434	    """Return the complete self-describing metadata block for one scaling row.
   435	
   436	    Parameters
   437	    ----------
   438	    grid, component, resolution, n_levels, precision, decomposition
   439	        Scientific descriptors of the case (caller-supplied).
   440	    n_ranks, n_gpus, devices_per_rank, cells_per_rank
   441	        Parallel layout.  ``devices_per_rank`` defaults to
   442	        ``jax.device_count() // jax.process_count()`` when omitted.
   443	    solver_variant, solver_residual, conservation_drift
   444	        Solver identity + convergence/conservation evidence.  The roadmap
   445	        forbids claiming a solver speedup without recording a residual /
   446	        conservation drift, so these belong in the record.
   447	    scaling_kind
   448	        ``"weak"`` | ``"strong"`` | ``"throughput"`` | ``None`` — keeps
   449	        weak/strong normalization (and single-device throughput-vs-size
   450	        sweeps, which are NOT device-count scaling) from being conflated
   451	        downstream.
   452	    transport
   453	        Halo/collective fabric (see :data:`TRANSPORTS`).  ``None`` →
   454	        auto-resolved by :func:`resolve_transport`; route-A mpi4jax drivers
   455	        that pass ``n_ranks`` explicitly resolve correctly, route-B /
   456	        single-process SPMD auto-detect from the live process.
   457	    partition_metrics
   458	        MPAS / Voronoi partition-quality numbers when available
   459	        (``edge_cut``, ``owned_halo_ratio``, ``cells_per_rank_min/max``,
   460	        ``message_count``).
   461	    extra
   462	        Any other component-specific descriptors.
   463	    """
   464	    backend = detect_backend()
   465	    device_count = _jax_count("device_count", -1)
   466	    process_count = _jax_count("process_count", 1)
   467	    # Resolve the transport BEFORE gpu_direct_mode: host_staged_halo /
   468	    # gpu_direct_active are mpi4jax-halo semantics and must not fire on a
   469	    # route-B NCCL / xla-local / serial row (codex finding 2).
   470	    # ``n_ranks`` = number of MPI processes.  Default to the auto-detected
   471	    # process count so a single-process SPMD run (1 process, N GPUs) records
   472	    # ``n_ranks=1`` (truthful) while the device parallelism lives in
   473	    # ``n_gpus`` / ``device_count``.  Route-A drivers (1 GPU/rank) pass the
   474	    # real rank count explicitly.
   475	    if n_ranks is None:
   476	        n_ranks = process_count
   477	    if devices_per_rank is None and device_count > 0 and process_count > 0:
   478	        devices_per_rank = device_count // process_count
   479	    resolved_transport = resolve_transport(
   480	        transport,
   481	        n_ranks=int(n_ranks),
   482	        process_count=process_count,
   483	        device_count=device_count,
   484	        backend=backend,
   485	    )
   486	
   487	    md: dict[str, Any] = {
   488	        "schema_version": METADATA_SCHEMA_VERSION,
   489	        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
   490	        # --- scientific descriptors ---
   491	        "grid": grid,
   492	        "component": component,
   493	        "resolution": resolution,
   494	        "n_levels": n_levels,
   495	        "precision": precision,
   496	        "decomposition": decomposition,
   497	        "solver_variant": solver_variant,
   498	        "solver_residual": solver_residual,
   499	        "conservation_drift": conservation_drift,
   500	        "scaling_kind": scaling_kind,
   501	        # --- parallel layout ---
   502	        "n_ranks": n_ranks,
   503	        "n_gpus": n_gpus,
   504	        "device_count": device_count,
   505	        "process_count": process_count,
   506	        "devices_per_rank": devices_per_rank,
   507	        "cells_per_rank": cells_per_rank,
   508	        # --- runtime facts (auto-detected) ---
   509	        "backend": backend,
   510	        "precision_knobs": precision_knobs(),
   511	        "transport": resolved_transport,
   512	        "virtual_cpu_devices": detect_virtual_cpu_devices(backend),
   513	        "launcher": detect_launcher(),
   514	        "hostname": os.environ.get("HOSTNAME")
   515	        or os.environ.get("SLURMD_NODENAME", ""),
   516	        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
   517	        "git_sha": git_sha(),
   518	    }
   519	    md.update(gpu_direct_mode(backend, transport=resolved_transport))
   520	    if partition_metrics:
   521	        md["partition_metrics"] = partition_metrics
   522	    if extra:
   523	        md["extra"] = extra
   524	    return md
   525	
   526	
   527	def validate_scaling_metadata(
   528	    md: dict[str, Any], *, strict: bool = True
   529	) -> list[str]:
   530	    """Return the list of comparability problems (empty = self-describing).
   531	
   532	    A REQUIRED key that is ``None``/``""`` and a PRESENT key that is absent
   533	    both count.  Raises ``ValueError`` when ``strict`` and any problem exists —
   534	    benchmark hygiene fail-fast, so a record that cannot be compared is never
   535	    silently written.  (``decomposition == "none"`` and ``solver_residual is
   536	    None`` are legal and do NOT count; ``0``/``False`` are legal values.)
   537	    """
   538	    missing = [k for k in REQUIRED_KEYS if _is_empty(md.get(k))]
   539	    absent = [f"{k}(absent)" for k in PRESENT_KEYS if k not in md]
   540	    problems = missing + absent
   541	    if strict and problems:
   542	        raise ValueError(
   543	            f"scaling metadata not self-describing: {problems}; a record that "
   544	            "cannot be compared must not be written (roadmap benchmark hygiene)."
   545	        )
   546	    return problems
   547	
   548	
   549	def annotate_incomplete(md: dict[str, Any], *, warn: bool = True) -> dict[str, Any]:
   550	    """Flag (do NOT discard) an incomplete record at write time.
   551	
   552	    A benchmark record is built AFTER an expensive run has already completed,
   553	    so a hard raise here would throw away real data.  Instead: validate
   554	    non-strictly, and if the record is not self-describing, embed the problem
   555	    list under ``md["_incomplete"]`` and emit a ``RuntimeWarning`` so the row
   556	    is LOUDLY flagged and a downstream aggregator can skip/annotate it.
   557	    Returns ``md`` (mutated) for chaining.  Use this at JSON-write time; use
   558	    :func:`validate_scaling_metadata` (strict) where aborting is acceptable.
   559	    """
   560	    problems = validate_scaling_metadata(md, strict=False)
   561	    if problems:
   562	        md["_incomplete"] = problems
   563	        if warn:
   564	            import warnings
   565	
   566	            warnings.warn(
   567	                f"scaling record not fully self-describing: {problems}",
   568	                RuntimeWarning,
   569	                stacklevel=2,
   570	            )
   571	    return md
   572	
   573	
   574	def tidy_throughput_fields(
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-12-# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-14-# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-43-  NODES=$(( NP / 4 ))
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-44-  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-45-  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-46-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:47:    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-48-      --multicontroller --n-devices "$NP" \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-49-      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-53-echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-54-for NP in 8 16 32; do
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-16-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-17-#   COMPOUNDING scale-out law (each 4x costs 1.5-1.9x): 18.7-23.6 ms
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-18-#   SATURATING (cost flattens past 32 devices):          ~12.5 ms
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-19-# Either outcome discriminates; protocol byte-matched to the s9 ladder
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:20:# (f32, sfc, --reorder-for 128, steps 12 / warmup 3).
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-21-set -uo pipefail
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-22-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-23-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-24-export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-30-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-31-rc=0
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-32-JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-33-    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:34:  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-35-    --multicontroller --n-devices 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-36-    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:37:    --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-38-    --out "$OUTDIR/np128.jsonl" || { echo "np128 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-39-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-40-import json,math,sys
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-41-try:
--
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-25-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-26-rc=0
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-27-JAX_ENABLE_X64=0 srun --ntasks=192 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-28-    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:29:  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-30-    --multicontroller --n-devices 192 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-31-    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:32:    --partition-method sfc --reorder-for 192 \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-33-    --out "$OUTDIR/np192.jsonl" || { echo "np192 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-34-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-35-import json,math,sys
scripts/cluster/scaling_levante/mpas_s10_192.sbatch-36-try:
--
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-16-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-17-#   COMPOUNDING scale-out law (each 4x costs 1.5-1.9x): 18.7-23.6 ms
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-18-#   SATURATING (cost flattens past 32 devices):          ~12.5 ms
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-19-# Either outcome discriminates; protocol byte-matched to the s9 ladder
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:20:# (f32, sfc, --reorder-for 128, steps 12 / warmup 3).
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-21-set -uo pipefail
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-22-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-23-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-24-export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
--
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-30-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-31-rc=0
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-32-JAX_ENABLE_X64=0 srun --ntasks=192 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-33-    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:34:  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-35-    --multicontroller --n-devices 192 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-36-    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:37:    --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-38-    --out "$OUTDIR/np192.jsonl" || { echo "np192 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-39-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-40-import json,math,sys
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-41-try:
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-65-  JAX_ENABLE_X64=0 srun --nodes=8 --ntasks=32 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-66-      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-67-      --job-name="arm_$1" \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-68-    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:69:    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-70-      --multicontroller --n-devices 32 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-71-      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:72:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-73-      --out "$OUTDIR/$1.jsonl"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-74-  s=$?
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-75-  echo "[$1] exit=$s epoch=$(date +%s.%N)"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-76-  return $s
--
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-26-OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_bound_j${SLURM_JOB_ID}}"
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-27-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-28-rc=0
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-29-JAX_ENABLE_X64=0 srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:30:  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-31-    --n-devices 1 --subdivision 6 --nlev 26 --steps 12 --warmup 3 \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch:32:    --lloyd 0 --partition-method sfc \
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-33-    --out "$OUTDIR/s6_nd1.jsonl" || { echo "s6_nd1 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-34-"$PY" -c "
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-35-import json,math,sys
scripts/cluster/scaling_levante/mpas_bound_base.sbatch-36-try:
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-60-RUN_OCEAN="${RUN_OCEAN:-1}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-61-RUN_MPAS="${RUN_MPAS:-1}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-62-# Lat-band lanes C/D task count (8 = 2 nodes; 16 = sbatch --nodes=4).
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-63-NP_LL="${NP_LL:-8}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:64:# Lane E device count; >6 pads the mesh via --reorder-for (even split).
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-65-NP_MPAS="${NP_MPAS:-6}"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-66-# Slurm job STEPS do not inherit the job's GPU allocation on all Slurm
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-67-# versions/configs — without an explicit step gres some tasks see zero
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-68-# devices (jax: "no supported devices found for platform CUDA"). Make every
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-193-# Lane E: icosahedral/MPAS multicontroller (6 procs = 2 nodes x 3 GPUs) —
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-194-# cell-partition reorder + ppermute halo (make_voronoi_sharded_step) over
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-195-# jax.distributed. np=6: nCells = 10*4^L+2 admits 1/2/3/6 even splits at
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-196-# every level. Smoke gate (parity+conservation) first, then the timed case.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:197:# Gate: tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-198-if [ "$RUN_MPAS" = "1" ]; then
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-199-    echo "=== LANE E: icosahedral MPAS multicontroller np=$NP_MPAS (L$ICO_LEVEL) ==="
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-200-    # >6 devices: 10*4^L+2 = 2*odd has no even split beyond 6 — pad the mesh
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:201:    # to the target count via --reorder-for (identical padded mesh per ladder).
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-202-    E_TPN=3; [ $((NP_MPAS % 4)) -eq 0 ] && E_TPN=4
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-203-    E_REORDER=""
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:204:    [ "$NP_MPAS" -gt 6 ] && E_REORDER="--reorder-for $NP_MPAS"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-205-    # shellcheck disable=SC2086
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-206-    srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:207:        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-208-        --multicontroller --n-devices "$NP_MPAS" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-209-        --subdivision 4 --nlev 8 --steps 4 --warmup 1 \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:210:        --partition-method sfc $E_REORDER \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-211-        --parity-gate --check-conservation \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-212-        --out "$OUTDIR/mpas_multicontroller/smoke.jsonl" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-213-    && srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:214:        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-215-        --multicontroller --n-devices "$NP_MPAS" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-216-        --subdivision "$ICO_LEVEL" --nlev "$NLEV" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-217-        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:218:        --partition-method sfc $E_REORDER \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:219:        --out "$OUTDIR/mpas_multicontroller/mpas_spmd_scaling.jsonl"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-220-    rc=$?
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-221-    [ "$rc" -ne 0 ] && { echo "LANE E FAILED rc=$rc"; rc_all=$rc; }
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-222-fi
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-223-

exec
/usr/bin/bash -lc "git log --all --oneline -S 'Reference census on the unrelaxed mesh' -- packages/core/legoesm/parallel/sharded_dynamics.py; git blame -L 1250,1326 packages/core/legoesm/parallel/sharded_dynamics.py; git log --all --oneline -S 'subdiv-8 sfc 12/14' -- packages/core/legoesm/parallel/sharded_dynamics.py packages/core/legoesm/parallel/voronoi_partition.py tests scripts; git log --all --format='%H%n%s%n%b%n---' -S 'subdiv-8 sfc 12/14' -- packages/core/legoesm/parallel/sharded_dynamics.py | head -n 500; git status --short scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch; git diff --check; git ls-files --error-unmatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch >/dev/null 2>&1; echo \"tracked="'$?"' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
exec
/usr/bin/bash -lc "git status --short scripts/bench/bench_voronoi_partition_methods.py tests/bench/test_bench_voronoi_partition_methods.py scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch; git diff --check; git log --all --oneline -S 'Reference census on the unrelaxed mesh' -- packages/core/legoesm/parallel/sharded_dynamics.py | head -n 40; git blame -L 1320,1327 packages/core/legoesm/parallel/sharded_dynamics.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 28210ms:
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py
?? scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
2a227a3c7 feat(parallel): spmd_schedule_cost — score a mesh split offline, before it costs GPU hours
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1320)         ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1321)         ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1322) 
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1323)     Reference census on the unrelaxed mesh, which any change here must still
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1324)     reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1325)     geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1326)     """
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1327)     from legoesm.parallel.voronoi_partition import (

exec
/usr/bin/bash -lc "rg -n -F -C 5 'subdiv-8 sfc 12/14' . -g '"'!*.pyc'"' -g '"'!*.npz'"' -g '"'!*.log'"' -g '"'!*.jsonl'"' -g '"'!*.json'"' || true; rg -n -F -C 5 '12/14 rounds' . -g '"'!*.pyc'"' -g '"'!*.npz'"' -g '"'!*.log'"' -g '"'!*.jsonl'"' -g '"'!*.json'"' || true; rg -n -F -C 5 '16/21' . -g '"'!*.pyc'"' -g '"'!*.npz'"' -g '"'!*.log'"' -g '"'!*.jsonl'"' -g '"'!*.json'"' || true; git show HEAD:packages/core/legoesm/parallel/sharded_dynamics.py | nl -ba | sed -n '1240,1330p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 1735ms:
./packages/core/legoesm/parallel/sharded_dynamics.py-1319-        ``resolved_method`` (concrete, never ``"auto"``),
./packages/core/legoesm/parallel/sharded_dynamics.py-1320-        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
./packages/core/legoesm/parallel/sharded_dynamics.py-1321-        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
./packages/core/legoesm/parallel/sharded_dynamics.py-1322-
./packages/core/legoesm/parallel/sharded_dynamics.py-1323-    Reference census on the unrelaxed mesh, which any change here must still
./packages/core/legoesm/parallel/sharded_dynamics.py:1324:    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
./packages/core/legoesm/parallel/sharded_dynamics.py-1325-    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
./packages/core/legoesm/parallel/sharded_dynamics.py-1326-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-1327-    from legoesm.parallel.voronoi_partition import (
./packages/core/legoesm/parallel/sharded_dynamics.py-1328-        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
./packages/core/legoesm/parallel/sharded_dynamics.py-1329-    )
--
./tests/parallel/test_spmd_schedule_cost.py-100-
./tests/parallel/test_spmd_schedule_cost.py-101-
./tests/parallel/test_spmd_schedule_cost.py-102-@pytest.mark.slow
./tests/parallel/test_spmd_schedule_cost.py-103-def test_sfc_beats_metis_and_geometric_on_rounds():
./tests/parallel/test_spmd_schedule_cost.py-104-    """The finding this function exists to make measurable. Census at
./tests/parallel/test_spmd_schedule_cost.py:105:    production sizes: subdiv-8 sfc 12/14 rounds at 64/128 devices vs metis
./tests/parallel/test_spmd_schedule_cost.py-106-    13/19, geometric 16/21. Same ordering here at the SMALLEST size that can
./tests/parallel/test_spmd_schedule_cost.py-107-    still tell the methods apart.
./tests/parallel/test_spmd_schedule_cost.py-108-
./tests/parallel/test_spmd_schedule_cost.py-109-    subdiv-4@8 and subdiv-5@8 score all three methods identically (7 rounds) —
./tests/parallel/test_spmd_schedule_cost.py-110-    too coarse to discriminate — so this uses subdiv-5@16 and asserts the
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-40-#     schedule there and the number is counterfactual.  Only meshes big
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-41-#     enough to keep cells/device above the threshold answer the question.
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-42-#
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-43-# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-44-# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:45:#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-46-#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-47-# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-48-# A scan that misses the known answer is a broken instrument, and its s10
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-49-# row must not be believed.
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-50-#
--
./packages/core/legoesm/parallel/sharded_dynamics.py-1319-        ``resolved_method`` (concrete, never ``"auto"``),
./packages/core/legoesm/parallel/sharded_dynamics.py-1320-        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
./packages/core/legoesm/parallel/sharded_dynamics.py-1321-        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
./packages/core/legoesm/parallel/sharded_dynamics.py-1322-
./packages/core/legoesm/parallel/sharded_dynamics.py-1323-    Reference census on the unrelaxed mesh, which any change here must still
./packages/core/legoesm/parallel/sharded_dynamics.py:1324:    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
./packages/core/legoesm/parallel/sharded_dynamics.py-1325-    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
./packages/core/legoesm/parallel/sharded_dynamics.py-1326-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-1327-    from legoesm.parallel.voronoi_partition import (
./packages/core/legoesm/parallel/sharded_dynamics.py-1328-        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
./packages/core/legoesm/parallel/sharded_dynamics.py-1329-    )
--
./tests/parallel/test_spmd_schedule_cost.py-100-
./tests/parallel/test_spmd_schedule_cost.py-101-
./tests/parallel/test_spmd_schedule_cost.py-102-@pytest.mark.slow
./tests/parallel/test_spmd_schedule_cost.py-103-def test_sfc_beats_metis_and_geometric_on_rounds():
./tests/parallel/test_spmd_schedule_cost.py-104-    """The finding this function exists to make measurable. Census at
./tests/parallel/test_spmd_schedule_cost.py:105:    production sizes: subdiv-8 sfc 12/14 rounds at 64/128 devices vs metis
./tests/parallel/test_spmd_schedule_cost.py-106-    13/19, geometric 16/21. Same ordering here at the SMALLEST size that can
./tests/parallel/test_spmd_schedule_cost.py-107-    still tell the methods apart.
./tests/parallel/test_spmd_schedule_cost.py-108-
./tests/parallel/test_spmd_schedule_cost.py-109-    subdiv-4@8 and subdiv-5@8 score all three methods identically (7 rounds) —
./tests/parallel/test_spmd_schedule_cost.py-110-    too coarse to discriminate — so this uses subdiv-5@16 and asserts the
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-40-#     schedule there and the number is counterfactual.  Only meshes big
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-41-#     enough to keep cells/device above the threshold answer the question.
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-42-#
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-43-# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-44-# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:45:#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-46-#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-47-# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-48-# A scan that misses the known answer is a broken instrument, and its s10
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-49-# row must not be believed.
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch-50-#
--
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-318-| Ocean test cases on all grids | ✅ DONE |
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-319-| OMIP | ✅ DONE (iter-25: matrix-compatible CSV + results.txt, ``run_omip_cross_grid.sh`` wrapper; iter-49 relaxes the ocean collector to accept timeseries-only OMIP runs so ``run_ocean_test_matrix.py --cross-grid-plots-only`` picks them up) |
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-320-| Same colorbar/projection across grids | ✅ DONE (cartopy PlateCarrée + shared cmap) |
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-321-| Physical consistency vs reference papers | ⚠️ Williamson cases agree to machine precision; HS climatology disagrees structurally (documented in §2) |
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-322-| GPU / MPI efficiency | ✅ DONE (iter-28/29: per-test-case ``wall-time/day`` ranking + speedup factor in every ``comparison_summary.txt``) |
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md:323:| `/codex:adversarial-review` | ✅ DONE (23 review rounds: iter-5/6/7/16/21/26/27/32/36/37/39/42/43/44/46/47/49/50/52/53/54/58/59, all findings addressed) |
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-324-
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-325----
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-326-
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-327-## 5. Recommended next steps (post-Ralph)
./docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md-328-
--
./docs/performance/scaling/scaling.md-15-
./docs/performance/scaling/scaling.md-16-(C24 is excluded — resolution-specific JW-jet resonance documented in
./docs/performance/scaling/scaling.md-17-§3.b.  CPU "sps" is from the 8-thread default backend; "GPU sps" is
./docs/performance/scaling/scaling.md-18-the RTX 5090 Laptop with the activated `--scan-steps` knob.)
./docs/performance/scaling/scaling.md-19-
./docs/performance/scaling/scaling.md:20:**What's in iter-216/217**: BCW emits a `RuntimeWarning` if user
./docs/performance/scaling/scaling.md-21-pairs `--scan-steps>1` with `--grid icosahedral` (iter-212 measured
./docs/performance/scaling/scaling.md-22-−9 % at I5).  The iter-199 vector-halo fix is now pinned by three
./docs/performance/scaling/scaling.md-23-regression tests in
./docs/performance/scaling/scaling.md-24-``tests/atmosphere/hydrostatic/unit/test_primitive_eq.py::TestHydrostaticToFV3VectorHalo``
./docs/performance/scaling/scaling.md-25-(round-trip, IC divergence, first-step ps balance).
--
./tests/parallel/test_spmd_schedule_cost.py-101-
./tests/parallel/test_spmd_schedule_cost.py-102-@pytest.mark.slow
./tests/parallel/test_spmd_schedule_cost.py-103-def test_sfc_beats_metis_and_geometric_on_rounds():
./tests/parallel/test_spmd_schedule_cost.py-104-    """The finding this function exists to make measurable. Census at
./tests/parallel/test_spmd_schedule_cost.py-105-    production sizes: subdiv-8 sfc 12/14 rounds at 64/128 devices vs metis
./tests/parallel/test_spmd_schedule_cost.py:106:    13/19, geometric 16/21. Same ordering here at the SMALLEST size that can
./tests/parallel/test_spmd_schedule_cost.py-107-    still tell the methods apart.
./tests/parallel/test_spmd_schedule_cost.py-108-
./tests/parallel/test_spmd_schedule_cost.py-109-    subdiv-4@8 and subdiv-5@8 score all three methods identically (7 rounds) —
./tests/parallel/test_spmd_schedule_cost.py-110-    too coarse to discriminate — so this uses subdiv-5@16 and asserts the
./tests/parallel/test_spmd_schedule_cost.py-111-    scores actually differ before asserting their order. ~100 s, hence slow.
--
./packages/core/legoesm/parallel/sharded_dynamics.py-1320-        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
./packages/core/legoesm/parallel/sharded_dynamics.py-1321-        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
./packages/core/legoesm/parallel/sharded_dynamics.py-1322-
./packages/core/legoesm/parallel/sharded_dynamics.py-1323-    Reference census on the unrelaxed mesh, which any change here must still
./packages/core/legoesm/parallel/sharded_dynamics.py-1324-    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
./packages/core/legoesm/parallel/sharded_dynamics.py:1325:    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
./packages/core/legoesm/parallel/sharded_dynamics.py-1326-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-1327-    from legoesm.parallel.voronoi_partition import (
./packages/core/legoesm/parallel/sharded_dynamics.py-1328-        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
./packages/core/legoesm/parallel/sharded_dynamics.py-1329-    )
./packages/core/legoesm/parallel/sharded_dynamics.py-1330-
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

 succeeded in 105123ms:
2a227a3c7 feat(parallel): spmd_schedule_cost — score a mesh split offline, before it costs GPU hours
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1250) def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1251)                        already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1252)                        ppermute_cells_per_device_threshold=2_000):
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1253)     """How much halo communication one ownership choice costs, computed offline.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1254) 
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1255)     Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1256)     ROUNDS one halo exchange needs -- the sequential collective launches that
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1257)     dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1258)     no MPI, no benchmark job, so a split can be compared before it costs an
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1259)     allocation.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1260) 
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1261)     It calls the SAME builders production calls
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1262)     (:func:`_build_voronoi_partition_infra` then
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1263)     :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1264)     different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1265)     rounds where the real depth-3-plus-closure graph reports 12-14.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1266) 
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1267)     WHAT THE NUMBER IS NOT
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1268)     ----------------------
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1269)     * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1270)       ``n_rounds`` x (tendency evaluations per step), which depends on the
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1271)       configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1272)       is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1273)     * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1274)       ``_build_ppermute_schedule`` tries a finite set of greedy orders and
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1275)       keeps the best; equality is MEASURED (compare the returned
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1276)       ``max_degree``), never assumed.  Do not claim "the colouring is already
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1277)       optimal so only ownership can help" from this function.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1278)     * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1279)       cells/device is below ``ppermute_cells_per_device_threshold``, in which
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1280)       case there is no ppermute schedule and this number is counterfactual --
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1281)       see the returned ``production_strategy``.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1282) 
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1283)     MESH STATE -- the one thing that silently invalidates the score
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1284)     --------------------------------------------------------------
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1285)     Production does NOT reorder inside ``make_voronoi_sharded_step``; it
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1286)     consumes an already-reordered ``model.mesh``.  The scaling bench reorders
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1287)     ONCE for a ``reorder_target`` device count and then runs at a possibly
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1288)     DIFFERENT device count.  So pass what you actually have:
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1289) 
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1290)     * raw mesh, scoring a run at ``n_dev``: defaults are right.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1291)     * raw mesh, but the run reorders for a different target: pass
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1292)       ``reorder_target=<that target>``; the split is built for the target and
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1293)       scored at ``n_dev``.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1294)     * already-reordered mesh (what production holds): pass
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1295)       ``already_reordered=True``; ``method`` is then ignored and reported as
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1296)       ``"pre-reordered"``, because the ownership is already baked in.
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1297) 
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1298)     Parameters
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1299)     ----------
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1300)     mesh : VoronoiMesh
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1301)     n_dev : int
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1302)         Device count the run uses.  Must be >= 1.
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1303)     method : str
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1304)         Ownership for the reorder; ignored when *already_reordered*.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1305)     reorder_target : int | None
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1306)         Device count the reorder targets, when it differs from *n_dev*.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1307)     already_reordered : bool
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1308)     halo_depth : int
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1309)         Must match production (3) or the graph is a different graph.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1310)     ppermute_cells_per_device_threshold : int
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1311)         Mirror of the production auto-select threshold, only used to report
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1312)         ``production_strategy``.
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1313) 
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1314)     Returns
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1315)     -------
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1316)     dict
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1317)         ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1318)         it against), ``n_rounds_greedy``, ``coloring_method``,
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1319)         ``resolved_method`` (concrete, never ``"auto"``),
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1320)         ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1321)         ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1322) 
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1323)     Reference census on the unrelaxed mesh, which any change here must still
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1324)     reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
2a227a3c79 (Pierre Gentine 2026-08-07 10:53:57 +0200 1325)     geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
86842d5fb0 (Pierre Gentine 2026-08-07 10:27:29 +0200 1326)     """
2a227a3c7 feat(parallel): spmd_schedule_cost — score a mesh split offline, before it costs GPU hours
2a227a3c79b1985ebfad738be6973b31f792bb04
feat(parallel): spmd_schedule_cost — score a mesh split offline, before it costs GPU hours
Scores a Voronoi ownership by how many sequential halo exchanges (ppermute
rounds) one fill needs — the term that binds MPAS strong scaling above ~64
devices. Runs on a laptop: no GPU, no MPI, no benchmark job. It calls the SAME
builders production calls, because a re-derived lookalike answers a different
question: a 1-ring cellsOnEdge graph reports 8 rounds where the real
depth-3-plus-closure graph reports 12-14.

Two codex rounds. Round 1 refuted it as a general API; round 2 found two more
defects. All fixed:

  - Production does NOT reorder inside make_voronoi_sharded_step; it consumes an
    already-reordered model.mesh, and the bench reorders once for a target then
    runs at a different count. Added already_reordered=True and reorder_target=N
    so both real cases are expressible; always reordering scored a mesh no run
    uses.
  - Scoring a mesh padded for one device count at a device count that does not
    divide it now RAISES. The builder assigns residual entities to the last
    owner but excludes them from every owned block, so send indices can exceed a
    shard — a plausible-looking wrong number (reproduced at target=3, n_dev=4).
  - already_reordered no longer reports reorder_target=n_dev; the target that
    produced the mesh is not recoverable from it, so it reports None.
  - Returns max_degree, and the docstring states that n_rounds is NOT proven
    equal to it: the colouring is a best-of-finite-greedy-orders search. The
    claim "the colouring is already optimal so only ownership can help" was
    wrong and is now explicitly forbidden.
  - Returns production_strategy: production auto-selects allgather below a
    cells/device threshold, where a ppermute round count is counterfactual.
    None at n_dev==1, where production returns before choosing.
  - Returns resolved_method (never "auto") via a new shared
    resolve_sharding_partition_method, so the label cannot drift from the
    auto->sfc policy the reorder applies.
  - n_rounds is per HALO FILL; the per-step multiplier is the integrator's
    tendency-evaluation count, and MPAS defaults to ssp_rk54_scan, not SSP-RK3.
  - halo_depth is now one shared SPMD_HALO_DEPTH constant consumed by both the
    scorer and the production factory, replacing two independent 3s.
  - Rejects non-integer n_dev (int() silently truncated 3.9 to 3).

Tests: 13 fast + 1 slow. The method-ordering test moved to subdiv-5@16 —
subdiv-4@8 and subdiv-5@8 score all three methods identically at 7 rounds, so
it was vacuous — and asserts the scores differ before asserting their order.
already_reordered is proven by spying on the reorder function, not inferred from
an equal round count.

Reference census this must keep reproducing: subdiv-8 sfc 12/14 rounds at
64/128 devices, metis 13/19, geometric 16/21; subdiv-9 sfc 11/13, metis 14/18,
geometric 14/18.

Known limits: ~40-200 s per candidate at subdiv-5/6 and it did not finish at
subdiv-8@128, so it suits a census of a few candidates, not a search over many.
already_reordered is trust-based — there is no provenance check on the mesh.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016q4upjC5GViwutba4AFQUC

---
 M scripts/bench/bench_voronoi_partition_methods.py
 M tests/bench/test_bench_voronoi_partition_methods.py
?? scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
tracked=1

exec
/usr/bin/bash -lc "nl -ba scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch | sed -n '1,180p'; nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '1013,1027p'; nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '1204,1213p'; nl -ba packages/core/legoesm/grids/voronoi.py | sed -n '1140,1158p'; nl -ba packages/core/legoesm/grids/voronoi.py | sed -n '1243,1270p'; nl -ba packages/core/legoesm/grids/voronoi.py | sed -n '1300,1315p'; nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1331,1400p'; nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1706,1872p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 331ms:
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
    39	# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row.
    40	#     REFUTES it: gap <= 1 on every production row -> recolouring is at or
    41	#     within one round of the Vizing floor, so the 7-14 day ownership build
    42	#     is the only remaining path, and is then justified.
    43	# (c) WHY NOT CHEAPER: this IS the cheap test — no GPU, no MPI, no model
    44	#     step.  It cannot be shrunk further onto small meshes: measured
    45	#     2026-08-07, L2/L4 x {geometric,sfc} x nd 2-16 all report gap == 0,
    46	#     but EVERY one of those rows auto-selects the ALLGATHER strategy
    47	#     (cells/device below the threshold), so production runs no ppermute
    48	#     schedule there and the number is counterfactual.  Only meshes big
    49	#     enough to keep cells/device above the threshold answer the question.
    50	#     s6 lloyd=0 np8/np16 (2560 cells/device, genuinely ppermute) also gave
    51	#     gap == 0 on all three methods — rounds 7/7/6 at np8 and 13/10/10 at
    52	#     np16 for geometric/sfc/metis.  That is a real but SMALL working
    53	#     point; s8-s10 are the production ones.
    54	#
    55	# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
    56	# docstring carries a reference census on the unrelaxed (lloyd=0) mesh:
    57	#   subdiv-8  sfc 12/14 rounds at 64/128 devices, metis 13/19, geometric 16/21
    58	#   subdiv-9  sfc 11/13,                          metis 14/18, geometric 14/18
    59	# Arms 1-2 must REPRODUCE those before arm 3's unknown s10 number is quoted.
    60	# A scan that misses the known answer is a broken instrument, and its s10
    61	# row must not be believed.
    62	#
    63	# lloyd=0 throughout: it is what the reference census used AND the cache key
    64	# the prewarmed s8/s9/s10 meshes were written under.  It is the LABELLED
    65	# synthetic scaling mesh, recorded in every row's metadata so it can never be
    66	# read back as a production SCVT receipt.
    67	#
    68	# Arms run cheapest-first and each writes its own JSON, so a later arm that
    69	# runs out of time or memory cannot lose an earlier arm's result.  s10@128 is
    70	# last and is the one genuinely at risk: the scorer is known not to have
    71	# finished at subdiv-8@128 on a laptop, which is why this asks for 24 h.
    72	#
    73	# SUBMIT (from the repo root):
    74	#   sbatch scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch
    75	# ===========================================================================
    76	set -uo pipefail
    77	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    78	export JAX_PLATFORMS=cpu
    79	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    80	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    81	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    82	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    83	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    84	
    85	OUT=results/a1/mpas_schedule_cost
    86	mkdir -p "$OUT"
    87	BENCH=scripts/bench/bench_voronoi_partition_methods.py
    88	
    89	# pymetis is present in .venv but was ABSENT from .venv-mpi for a while, and a
    90	# missing partitioner is reported as "unavailable" rather than substituted.
    91	# Say which python and whether metis is really there, so a two-method table
    92	# cannot be misread as a three-method one.
    93	echo "[scan] python=$PY"
    94	"$PY" -c "import pymetis; print('[scan] pymetis OK')" \
    95	  || echo "[scan] pymetis MISSING -> metis rows will report unavailable"
    96	
    97	run_arm () {  # $1=level  $2=rank-counts  $3=label
    98	  echo "=== [scan] arm $3: subdiv-$1 nd=$2 (lloyd=0) ==="
    99	  date -u +"[scan] start %Y-%m-%dT%H:%M:%SZ"
   100	  "$PY" "$BENCH" \
   101	      --subdivision "$1" --rank-counts "$2" --lloyd 0 \
   102	      --methods geometric,sfc,metis --schedule-cost \
   103	      --out "$OUT/schedule_cost_s$1.json"
   104	  echo "[scan] arm $3 exit=$?"
   105	  date -u +"[scan] end   %Y-%m-%dT%H:%M:%SZ"
   106	}
   107	
   108	# Arms 1-2: KNOWN answers (the docstring census) — instrument validation.
   109	run_arm 8  64,128  "1/3 VALIDATION s8"
   110	run_arm 9  64,128  "2/3 VALIDATION s9"
   111	# Arm 3: the record working point (s10@128 = 18.20 ms = 14.98 GCells/s), unknown.
   112	run_arm 10 128     "3/3 UNKNOWN s10"
   113	
   114	echo "SCAN_DONE"
  1013	def resolve_sharding_partition_method(method: str) -> str:
  1014	    """Concrete ownership for the SPMD/ppermute path (``auto`` -> ``sfc``).
  1015	
  1016	    Separate from :func:`resolve_partition_method`, whose ``auto`` prefers
  1017	    METIS: METIS minimizes edge CUT, while this path is bound by the number of
  1018	    sequential halo exchanges, and measured they move oppositely (subdiv-8 at
  1019	    128 devices: sfc 14 rounds, metis 19).  ONE definition, so a scorer that
  1020	    reports which ownership ran cannot drift from what the reorder does.
  1021	    """
  1022	    if method == "auto":
  1023	        return "sfc"
  1024	    return resolve_partition_method(method)
  1025	
  1026	
  1027	def reorder_voronoi_for_sharding(
  1204	        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
  1205	                             else reorder_1d(mesh.subgrid_topo_stddev,
  1206	                                             cell_perm)),
  1207	        land_frac=(None if mesh.land_frac is None
  1208	                   else reorder_1d(mesh.land_frac, cell_perm)),
  1209	    )
  1210	
  1211	    # --- Pad so that nCells and nEdges are divisible by n_devices ---
  1212	    return _pad_voronoi_for_sharding(reordered, n_devices)
  1140	
  1141	
  1142	def _voronoi_cache_path(
  1143	    subdivision_level: int, radius: float, lloyd_iterations: int, omega: float
  1144	) -> str:
  1145	    """Deterministic cache filename keyed on every parameter that changes the mesh.
  1146	
  1147	    Includes the active x64 flag because the stored dtypes (and a fresh rebuild)
  1148	    depend on it — a float32-built mesh must never be served to an x64 run.
  1149	    ``radius``/``omega`` use ``repr`` (round-trippable for Python floats).
  1150	    """
  1151	    x64 = bool(jax.config.read("jax_enable_x64"))
  1152	    name = (
  1153	        f"scvt_v{_MESH_CACHE_VERSION}_lvl{subdivision_level}"
  1154	        f"_r{radius!r}_omega{omega!r}_lloyd{lloyd_iterations}"
  1155	        f"_x64{int(x64)}.npz"
  1156	    )
  1157	    return os.path.join(_voronoi_cache_dir(), name)
  1158	
  1243	def create_voronoi_mesh(
  1244	    subdivision_level: int,
  1245	    radius: float = constants.R_earth,
  1246	    lloyd_iterations: int = 50,
  1247	    omega: float = constants.Omega,
  1248	    density_fn=None,
  1249	) -> VoronoiMesh:
  1250	    """Create a centroidal Voronoi tessellation (SCVT) on the sphere.
  1251	
  1252	    Starts from an icosahedral triangulation, bisects to the desired
  1253	    level, applies Lloyd relaxation, then builds the full MPAS-compatible
  1254	    mesh with all connectivity and geometric arrays.
  1255	
  1256	    Parameters
  1257	    ----------
  1258	    subdivision_level : int
  1259	        Number of bisection levels. nCells = 10*4^level + 2.
  1260	        level=3: 642 cells, level=4: 2562, level=5: 10242.
  1261	    radius : float
  1262	        Sphere radius [m]. Default: Earth radius.
  1263	    lloyd_iterations : int
  1264	        Number of Lloyd relaxation iterations. Default: 50.
  1265	    omega : float
  1266	        Rotation rate [rad/s]. Default: Earth rotation.
  1267	    density_fn : callable(lat, lon) -> float, optional
  1268	        Relative mesh-density function (``lat``, ``lon`` in radians; larger =>
  1269	        finer cells).  When provided, the Lloyd relaxation is **density-weighted**
  1270	        (Du–Faber–Gunzburger SCVT): generators concentrate where the density is
  1300	
  1301	    # Disk cache: a uniform SCVT mesh is deterministic in these args, so skip the
  1302	    # expensive rebuild on a hit.  density_fn meshes are NOT cached (a callable
  1303	    # has no stable key); the env switch lets a run force a fresh build.
  1304	    use_cache = density_fn is None and not _voronoi_cache_disabled()
  1305	    cache_path = None
  1306	    if use_cache:
  1307	        cache_path = _voronoi_cache_path(
  1308	            subdivision_level, radius, lloyd_iterations, omega)
  1309	        cached = _load_voronoi_cache(cache_path)
  1310	        if cached is not None:
  1311	            return cached
  1312	
  1313	    _lock_path = None
  1314	    if _big:
  1315	        if not use_cache:
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

exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '100,150p'; rg -n -C 3 \"def resolve_partition_method\" packages/core/legoesm/parallel/voronoi_partition.py; nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '210,290p'; nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1910,1979p'; nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '2090,2110p'; nl -ba tests/bench/test_bench_voronoi_partition_methods.py | sed -n '154,285p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 260ms:
   100	
   101	        return True
   102	    except Exception:
   103	        return False
   104	
   105	
   106	def partition_quality(mesh, cell_owner: np.ndarray, n_ranks: int,
   107	                      halo_depth: int = 2) -> dict:
   108	    """Exact partition-quality metrics from a global owner array.
   109	
   110	    Serial enumeration of every rank (no MPI): the same halo construction
   111	    the runtime uses (``compute_halo_cells``), so the reported halo sizes
   112	    are the runtime's, not an estimate.
   113	    """
   114	    from legoesm.parallel.voronoi_partition import compute_halo_cells
   115	
   116	    n_cells = int(mesh.nCells)
   117	    # Correctness: every cell owned exactly once, owners in range, no
   118	    # empty rank (an empty rank would silently deflate the halo/owned
   119	    # ratio through the max(counts, 1) guard).
   120	    if n_cells == 0 or cell_owner.size == 0:
   121	        raise AssertionError("empty mesh / owner array")
   122	    if cell_owner.shape != (n_cells,):
   123	        raise AssertionError(f"owner shape {cell_owner.shape} != ({n_cells},)")
   124	    if cell_owner.min() < 0 or cell_owner.max() >= n_ranks:
   125	        raise AssertionError("owner out of range")
   126	    counts = np.bincount(cell_owner, minlength=n_ranks).astype(float)
   127	    if int(counts.sum()) != n_cells:
   128	        raise AssertionError("ownership does not cover the mesh")
   129	    if counts.min() <= 0:
   130	        raise AssertionError(
   131	            f"empty rank in partition (counts.min()={counts.min():.0f}) — "
   132	            f"a skipped rank corrupts every per-rank metric")
   133	
   134	    # Edge cut: edges whose two cells have different owners.
   135	    c1, c2 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
   136	    valid = (c1 >= 0) & (c2 >= 0)
   137	    edge_cut = int((cell_owner[c1[valid]] != cell_owner[c2[valid]]).sum())
   138	
   139	    halo_sizes = []
   140	    neighbor_counts = []
   141	    cells_on_cell = np.asarray(mesh.cellsOnCell)
   142	    max_edges = int(cells_on_cell.shape[0]) if cells_on_cell.ndim == 2 else 0
   143	    for r in range(n_ranks):
   144	        halo = compute_halo_cells(
   145	            cell_owner, mesh.cellsOnCell, mesh.maxEdges, r, halo_depth)
   146	        halo_sizes.append(len(halo))
   147	        neighbor_counts.append(
   148	            len(set(int(cell_owner[c]) for c in halo) - {r}))
   149	    _ = max_edges
   150	    halo_sizes = np.array(halo_sizes, dtype=float)
57-    return importlib.util.find_spec("pymetis") is not None
58-
59-
60:def resolve_partition_method(method: str) -> str:
61-    """Resolve a partition method, expanding ``"auto"`` by available capability.
62-
63-    ``"auto"`` (the default) selects ``"metis"`` when ``pymetis`` is importable —
   210	# Partitioners
   211	# ============================================================================
   212	
   213	def partition_cells_geometric(mesh: VoronoiMesh, n_ranks: int) -> np.ndarray:
   214	    """Partition cells via Recursive Coordinate Bisection (RCB).
   215	
   216	    Uses cell-center Cartesian coordinates on the unit sphere.
   217	
   218	    Parameters
   219	    ----------
   220	    mesh : VoronoiMesh
   221	    n_ranks : int
   222	
   223	    Returns
   224	    -------
   225	    cell_owner : np.ndarray, shape (nCells,), dtype int32
   226	        ``cell_owner[c]`` is the rank that owns cell ``c``.
   227	    """
   228	    coords = np.stack([
   229	        np.asarray(mesh.xCell) / mesh.radius,
   230	        np.asarray(mesh.yCell) / mesh.radius,
   231	        np.asarray(mesh.zCell) / mesh.radius,
   232	    ], axis=1)
   233	    return _rcb(coords, n_ranks)
   234	
   235	
   236	def _rcb(coords: np.ndarray, n_ranks: int) -> np.ndarray:
   237	    """Recursive Coordinate Bisection on a point cloud."""
   238	    n = len(coords)
   239	    if n_ranks <= 1 or n <= 1:
   240	        return np.zeros(n, dtype=np.int32)
   241	
   242	    axis = int(np.argmax(np.ptp(coords, axis=0)))
   243	    order = np.argsort(coords[:, axis])
   244	
   245	    n_left_ranks = n_ranks // 2
   246	    n_right_ranks = n_ranks - n_left_ranks
   247	    split = max(1, min(n - 1, n * n_left_ranks // n_ranks))
   248	
   249	    left, right = order[:split], order[split:]
   250	    result = np.empty(n, dtype=np.int32)
   251	    result[left] = _rcb(coords[left], n_left_ranks)
   252	    result[right] = _rcb(coords[right], n_right_ranks) + n_left_ranks
   253	    return result
   254	
   255	
   256	def partition_cells_metis(mesh: VoronoiMesh, n_ranks: int) -> np.ndarray:
   257	    """Partition cells via METIS k-way graph partitioning.
   258	
   259	    Requires the ``pymetis`` package.
   260	
   261	    Parameters
   262	    ----------
   263	    mesh : VoronoiMesh
   264	    n_ranks : int
   265	
   266	    Returns
   267	    -------
   268	    cell_owner : np.ndarray, shape (nCells,), dtype int32
   269	    """
   270	    try:
   271	        import pymetis
   272	    except ImportError as exc:
   273	        raise ImportError(
   274	            "METIS partitioning requires pymetis.  "
   275	            "Install with: pip install pymetis"
   276	        ) from exc
   277	
   278	    coc = np.asarray(mesh.cellsOnCell)
   279	    nec = np.asarray(mesh.nEdgesOnCell)
   280	    adjacency = []
   281	    for c in range(mesh.nCells):
   282	        nbrs = [int(coc[k, c]) for k in range(int(nec[c])) if coc[k, c] >= 0]
   283	        adjacency.append(np.array(nbrs, dtype=np.int32))
   284	
   285	    _, membership = pymetis.part_graph(n_ranks, adjacency=adjacency)
   286	    return np.array(membership, dtype=np.int32)
   287	
   288	
   289	def _hilbert_xy2d(order: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
   290	    """Hilbert-curve distance ``d`` for integer grid coords ``(x, y)``.
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
  2090	    # +1 garbage slot for padded scatter targets (trimmed at the end):
  2091	    # schedule rows are padded to the round's max halo count, and padding
  2092	    # entries target position max_lc / max_le.
  2093	    cell_local = jnp.pad(cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)))
  2094	    u_local = jnp.pad(u_shard, ((0, max_le + 1 - edges_per), (0, 0)))
  2095	
  2096	    for r, (sc, rc, se, re) in enumerate(halo_sl):
  2097	        send_c = cell_pack[sc[0]]             # (hc_r, W)
  2098	        send_e = u_shard[se[0]]               # (he_r, nlev)
  2099	        send_c_flat = send_c.ravel()
  2100	        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
  2101	        recv_packed = jax.lax.ppermute(
  2102	            send_packed, "device", perm=ppermute_perms[r])
  2103	        split_at = send_c_flat.shape[0]       # static
  2104	        recv_c = recv_packed[:split_at].reshape(send_c.shape)
  2105	        recv_e = recv_packed[split_at:].reshape(send_e.shape)
  2106	        cell_local = cell_local.at[rc[0]].set(recv_c)
  2107	        u_local = u_local.at[re[0]].set(recv_e)
  2108	
  2109	    return cell_local[:max_lc], u_local[:max_le]
  2110	
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
   170	    "n_rounds, max_degree, gap, headroom, proven",
   171	    [
   172	        # Provably optimal: no proper edge colouring beats max_degree.
   173	        (12, 12, 0, 0, True),
   174	        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
   175	        # buys nothing provable — this is the case that a naive
   176	        # "gap > 0 means recolour" rule would over-claim.
   177	        (13, 12, 1, 0, False),
   178	        # Only here is recolouring guaranteed to remove rounds, and at most
   179	        # gap-1 of them.
   180	        (14, 10, 4, 3, False),
   181	    ])
   182	def test_coloring_gap_and_headroom_are_derived_not_assumed(
   183	        monkeypatch, n_rounds, max_degree, gap, headroom, proven):
   184	    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.
   185	
   186	    Non-vacuity, the hard way: on every mesh small enough to test quickly the
   187	    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
   188	    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
   189	    time), so a real-mesh assertion cannot tell a correct subtraction from a
   190	    hardcoded ``0``; that exact mutation passed the first version of this
   191	    test.  Stubbing the production scorer with KNOWN values is what makes
   192	    the assertion able to fail.
   193	
   194	    The ``gap == 1`` row is the one that matters: the schedule is a proper
   195	    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
   196	    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
   197	    indistinguishable from optimal (Class 2), and claiming recolouring
   198	    headroom there would be an over-claim.
   199	    """
   200	    stub = {
   201	        "n_rounds": n_rounds, "max_degree": max_degree,
   202	        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
   203	        "resolved_method": "geometric", "halo_depth": 3,
   204	        "cells_per_device": 99_999, "production_strategy": "ppermute",
   205	    }
   206	    import legoesm.parallel.sharded_dynamics as sd
   207	    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
   208	
   209	    sc = mod.schedule_cost_row(object(), "geometric", 8)
   210	    assert sc["coloring_gap"] == gap
   211	    assert sc["coloring_headroom_rounds"] == headroom
   212	    assert sc["coloring_optimal_proven"] is proven
   213	
   214	
   215	def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
   216	    """Real-mesh sanity on the bound itself: a proper edge colouring can
   217	    never use fewer rounds than the graph's max degree, and the multi-start
   218	    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
   219	    fires, ``max_degree`` is not the degree of the graph being coloured and
   220	    every gap-based conclusion built on it is void."""
   221	    for method in ("geometric", "sfc"):
   222	        for n_ranks in (2, 4, 8):
   223	            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
   224	            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
   225	                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
   226	                f"max_degree={sc['max_degree']}")
   227	
   228	
   229	def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
   230	    """The schedule is scored at the SPMD production halo depth, NOT this
   231	    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
   232	    a different graph and quietly report the wrong lane's cost."""
   233	    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
   234	
   235	    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
   236	    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
   237	
   238	
   239	def test_schedule_cost_matches_the_production_scorer_exactly():
   240	    """Lock: the wrapper reports what the production scorer returns — it is
   241	    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
   242	    reports 8 rounds where the real depth-3 graph reports 12-14)."""
   243	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   244	
   245	    mesh = _mesh()
   246	    ref = spmd_schedule_cost(mesh, 2, method="sfc")
   247	    sc = mod.schedule_cost_row(mesh, "sfc", 2)
   248	    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
   249	                "coloring_method", "resolved_method", "cells_per_device",
   250	                "production_strategy"):
   251	        assert sc[key] == ref[key], key
   252	
   253	
   254	def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
   255	        tmp_path, monkeypatch):
   256	    """Off by default (it is the expensive layer); on, every scored row
   257	    carries the schedule block and the run records that it ran."""
   258	    out = tmp_path / "off.json"
   259	    monkeypatch.setattr(sys, "argv", [
   260	        "bench", "--subdivision", "2", "--rank-counts", "2",
   261	        "--methods", "geometric", "--out", str(out)])
   262	    assert mod.main() == 0
   263	    payload = json.loads(out.read_text())
   264	    assert "schedule" not in payload["rows"][0]
   265	    assert payload["metadata"]["extra"]["schedule_cost"] is False
   266	
   267	    out2 = tmp_path / "on.json"
   268	    monkeypatch.setattr(sys, "argv", [
   269	        "bench", "--subdivision", "2", "--rank-counts", "2",
   270	        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
   271	    assert mod.main() == 0
   272	    payload2 = json.loads(out2.read_text())
   273	    row = payload2["rows"][0]
   274	    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
   275	    assert payload2["metadata"]["extra"]["schedule_cost"] is True
   276	    # Mesh provenance: a lloyd=0 synthetic mesh must never be readable as a
   277	    # production SCVT receipt.
   278	    assert payload2["metadata"]["extra"]["lloyd_iterations"] == 50

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/voronoi_partition.py | sed -n '48,80p'; nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '60,105p'; nl -ba scripts/bench/bench_voronoi_partition_methods.py | sed -n '328,352p'; nl -ba scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch | sed -n '12,54p'; nl -ba scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch | sed -n '59,76p'; nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '304,317p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 258ms:
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
    60	   (atmosphere SPMD) — one method per launch, same case otherwise
    61	   (controlled comparison).
    62	
    63	Guards: methods that are unavailable (``metis`` without ``pymetis``) are
    64	reported as ``"unavailable"`` — never silently substituted, so a table
    65	column can never claim METIS numbers that actually came from the RCB
    66	fallback.  Partition CORRECTNESS is asserted per row (every cell owned by
    67	exactly one rank; owner range valid) before any metric is recorded.
    68	
    69	Run:
    70	  python scripts/bench/bench_voronoi_partition_methods.py \
    71	      --subdivision 6 --rank-counts 2,4,8,16 --out results/partition_quality.json
    72	
    73	  # + the SPMD schedule depth (minutes to hours at subdiv>=8 — batch it):
    74	  python scripts/bench/bench_voronoi_partition_methods.py \
    75	      --subdivision 9 --rank-counts 64,128 --schedule-cost \
    76	      --out results/a1/schedule_cost_s9.json
    77	"""
    78	from __future__ import annotations
    79	
    80	import argparse
    81	import json
    82	import os
    83	import sys
    84	from pathlib import Path
    85	
    86	import numpy as np
    87	
    88	sys.path.insert(0, str(Path(__file__).resolve().parent))
    89	
    90	from metadata import annotate_incomplete, scaling_metadata  # noqa: E402
    91	
    92	METHODS = ("geometric", "sfc", "metis")
    93	
    94	
    95	def method_available(method: str) -> bool:
    96	    if method != "metis":
    97	        return True
    98	    try:
    99	        import pymetis  # noqa: F401
   100	
   101	        return True
   102	    except Exception:
   103	        return False
   104	
   105	
   328	    payload = {
   329	        "rows": rows,
   330	        "auto_resolves_to": resolve_partition_method("auto"),
   331	        "step_time_pointer": (
   332	            "step-time per method: bench_ocean_mpas_scaling.py / "
   333	            "bench_mpas_spmd_scaling.py --partition-method <m> (one method "
   334	            "per launch, same case otherwise)"),
   335	        "metadata": annotate_incomplete(scaling_metadata(
   336	            grid="voronoi",
   337	            component="partitioning",
   338	            resolution=f"L{args.subdivision}",
   339	            n_levels=0,
   340	            precision="n/a",
   341	            decomposition="cell_partition",
   342	            solver_variant="n/a",
   343	            scaling_kind="partition-quality",
   344	            transport="none",
   345	            extra={"rank_counts": rank_counts, "methods": methods,
   346	                   "halo_depth": args.halo_depth,
   347	                   "schedule_cost": bool(args.schedule_cost),
   348	                   "lloyd_iterations": args.lloyd},
   349	        )),
   350	    }
   351	    outdir = os.path.dirname(args.out)
   352	    if outdir:
    12	# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
    13	# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
    14	# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
    15	# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
    16	# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
    17	# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
    18	# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
    19	#
    20	# Falsifiability, written BEFORE submit:
    21	#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
    22	#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
    23	#             matched cells/GPU stay well above 1 (prior draft saw
    24	#             1.80/1.35/1.41 on the CONFOUNDED pairs)
    25	#   REFUTE  : ratios collapse toward ~1.0 -> the draft's "term" was the
    26	#             Lloyd-mesh confound, and MPAS-GPU weak scaling is near
    27	#             ideal at matched tile.
    28	# Mesh: prewarmed into LEGOESM_MESH_CACHE_DIR (subdiv-8 is below the
    29	# big-mesh refuse threshold, so a cache miss falls back to in-process
    30	# builds — slower, still correct).
    31	set -uo pipefail
    32	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    33	export JAX_PLATFORMS=cuda,cpu
    34	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    35	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    36	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    37	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    38	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    39	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s8_l0_j${SLURM_JOB_ID}}"
    40	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    41	rc=0
    42	for NP in 8 16 32; do
    43	  NODES=$(( NP / 4 ))
    44	  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
    45	  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
    46	      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    47	    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    48	      --multicontroller --n-devices "$NP" \
    49	      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
    50	      --partition-method sfc --reorder-for 128 \
    51	      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
    52	done
    53	echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
    54	for NP in 8 16 32; do
    59	
    60	run_arm () { # tag  (one 32-GPU replica on 8 disjoint nodes)
    61	  echo "[$1] launch epoch=$(date +%s.%N)"
    62	  # rank 0 prints its step's nodelist BEFORE exec'ing the bench — the
    63	  # per-replica disjointness + coordinator-host evidence codex asked for
    64	  # ("$0" inside bash -c is $PY, passed as the first post-script arg).
    65	  JAX_ENABLE_X64=0 srun --nodes=8 --ntasks=32 --ntasks-per-node=4 \
    66	      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
    67	      --job-name="arm_$1" \
    68	    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
    69	    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    70	      --multicontroller --n-devices 32 \
    71	      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
    72	      --partition-method sfc --reorder-for 128 \
    73	      --out "$OUTDIR/$1.jsonl"
    74	  s=$?
    75	  echo "[$1] exit=$s epoch=$(date +%s.%N)"
    76	  return $s
   304	    dt = args.dt
   305	    if dt is None:
   306	        dt = max(600.0 * 4.0 ** (4 - args.subdivision), 30.0)
   307	
   308	    reorder_for = args.reorder_for if args.reorder_for is not None else nd
   309	    if reorder_for < nd:
   310	        raise SystemExit(
   311	            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
   312	            f"the ghost padding only guarantees divisibility for the "
   313	            f"partition target.")
   314	    mesh, model, s0, dev_config = build_model_and_state(
   315	        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
   316	        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
   317	

exec
/usr/bin/bash -lc "rg -n -C 3 -- '--n-devices 64|--n-devices \"\\"'$NP"|--n-devices '"\\"'$NP|--reorder-for 128'"' scripts/cluster/scaling_levante -g '*.sbatch' -g '*.slurm' -g '*.sh' | rg -C 6 'mpas|n-devices|reorder-for' | head -n 700" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 326ms:
rg: -g: No such file or directory (os error 2)
rg: *.sbatch: No such file or directory (os error 2)
rg: -g: No such file or directory (os error 2)
rg: *.slurm: No such file or directory (os error 2)
rg: -g: No such file or directory (os error 2)
rg: *.sh: No such file or directory (os error 2)
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch-47-  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=64 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch-48-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch-49-    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:50:      --multicontroller --n-devices 64 --mode strong \
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch-51-      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch-52-      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch-53-}
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-14-# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-45-  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-46-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-47-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:48:      --multicontroller --n-devices "$NP" \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-49-      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-53-echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-17-#   COMPOUNDING scale-out law (each 4x costs 1.5-1.9x): 18.7-23.6 ms
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-18-#   SATURATING (cost flattens past 32 devices):          ~12.5 ms
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-19-# Either outcome discriminates; protocol byte-matched to the s9 ladder
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:20:# (f32, sfc, --reorder-for 128, steps 12 / warmup 3).
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-21-set -uo pipefail
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-22-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-23-export JAX_PLATFORMS=cuda,cpu
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-34-  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-35-    --multicontroller --n-devices 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-36-    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:37:    --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-38-    --out "$OUTDIR/np128.jsonl" || { echo "np128 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-39-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-40-import json,math,sys
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-69-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-70-      --multicontroller --n-devices 32 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-71-      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:72:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-73-      --out "$OUTDIR/$1.jsonl"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-74-  s=$?
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-75-  echo "[$1] exit=$s epoch=$(date +%s.%N)"
--
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-17-#   COMPOUNDING scale-out law (each 4x costs 1.5-1.9x): 18.7-23.6 ms
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-18-#   SATURATING (cost flattens past 32 devices):          ~12.5 ms
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-19-# Either outcome discriminates; protocol byte-matched to the s9 ladder
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:20:# (f32, sfc, --reorder-for 128, steps 12 / warmup 3).
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-21-set -uo pipefail
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-22-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-23-export JAX_PLATFORMS=cuda,cpu
--
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-34-  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-35-    --multicontroller --n-devices 192 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-36-    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch:37:    --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-38-    --out "$OUTDIR/np192.jsonl" || { echo "np192 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-39-"$PY" -c "
scripts/cluster/scaling_levante/mpas_s10_192b.sbatch-40-import json,math,sys

codex
## Findings

- **BLOCKER: “validation” is only a comment.** The sbatch never compares JSON results with the census, never requires all three methods, and continues after a failed arm. `run_arm` logs `$?`, then returns the succeeding `date` status; the script has no `set -e` and proceeds to s10 regardless. A missing `pymetis` is explicitly tolerated, too. [mpas_schedule_cost_scan.sbatch:76](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:76) [mpas_schedule_cost_scan.sbatch:94](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:94) [mpas_schedule_cost_scan.sbatch:97](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:97) [mpas_schedule_cost_scan.sbatch:108](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:108)  
  As written, arms 1–2 do not validate anything and cannot gate quoting s10.

- **Your `gap <= 1 ⇒ ownership is the only path` conclusion is false.** `gap == 1` is inconclusive: a better colouring may remove one round (Class 1), or it may already be optimal (Class 2). The test correctly says this, but the batch’s decision rule still says it “REFUTES” recolouring and makes ownership the only path. [test_bench_voronoi_partition_methods.py:194](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:194) [mpas_schedule_cost_scan.sbatch:39](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:39)  
  Correct discriminator: `gap == 0` rules out a better **undirected edge-colouring of this graph**; `gap == 1` does not.

- **`coloring_headroom_rounds` is documented and printed backwards.** For `n_rounds=14, Δ=10`, Vizing guarantees a reduction of **at least** 3 rounds, while 4 remain possible. `max(0, gap-1)` is a lower bound on improvement, not the “MAXIMUM” improvement; the `<= 3` output is wrong. The test repeats the same error (“at most”). [bench_voronoi_partition_methods.py:194](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:194) [bench_voronoi_partition_methods.py:225](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:225) [bench_voronoi_partition_methods.py:317](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:317) [test_bench_voronoi_partition_methods.py:178](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:178)

- **(a)/(b): the bound is valid, but only for the implementation’s artificially symmetric graph.** `max_degree` is computed from the exact `comm_pairs` later coloured, and properness is asserted, so `gap == 0` genuinely proves optimality for that simple undirected graph. It is neither a cell-degree nor edge-degree mixup. [sharded_dynamics.py:1733](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1733) [sharded_dynamics.py:1815](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1815) [sharded_dynamics.py:1851](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1851) [sharded_dynamics.py:1864](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1864)  
  But `comm_pairs` is formed when either direction has a halo dependency, then every pair is forced into *both* ppermute directions—even when one send map is empty. [sharded_dynamics.py:1795](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1795) [sharded_dynamics.py:1923](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1923) [sharded_dynamics.py:1945](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1945)  
  Therefore Δ is not a lower bound for a redesigned *directed* schedule/packing that exploits one-way exchanges. “Only a new ownership objective remains” overclaims even at gap zero.

- **The provenance claim is false at row level.** `--lloyd` does reach mesh construction, and schedule scoring does not consume MPI `--halo-depth`; those two code paths are correct. [bench_voronoi_partition_methods.py:217](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:217) [bench_voronoi_partition_methods.py:279](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:279) [sharded_dynamics.py:1250](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1250)  
  But `lloyd_iterations` exists only once in enclosing metadata, not in each row/schedule block as the sbatch claims. Worse, the wrapper drops scorer provenance including `reorder_target` and `already_reordered`. A flattened/copied row cannot establish that it was raw, `lloyd=0`, or partitioned for the reported device count. [bench_voronoi_partition_methods.py:219](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:219) [bench_voronoi_partition_methods.py:345](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:345) [sharded_dynamics.py:1375](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1375)

- **The `--lloyd` test is vacuous.** It asserts the default value `50`; removing `lloyd_iterations=args.lloyd` would still pass. There is no `--lloyd 0` invocation or mesh-builder spy. [test_bench_voronoi_partition_methods.py:267](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:267)

- **(c): yes, the small even-device `rounds == n_dev - 1, gap == 0` cases are complete-graph saturation.** Δ then equals `n_dev - 1`, which is only possible for \(K_{n_dev}\). s6@16 escapes that regime because 13/10/10 is below 15; geometric is still very dense, but it is not complete-graph saturated. Also, “2560 cells/device” is wrong for the actual score: L6 has 40,962 cells and padding to 16 produces 40,976, hence **2561**. It remains safely above the 2,000 ppermute threshold. [voronoi.py:1259](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/voronoi.py:1259) [voronoi_partition.py:1211](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_partition.py:1211) [sharded_dynamics.py:1393](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1393)

- **(d): the invocation does statically match the docstring’s stated raw-mesh/default-target setup**—`--lloyd 0`, explicit method, then `spmd_schedule_cost(raw_mesh, n_dev, method=...)`, whose default target is `n_dev`. [mpas_schedule_cost_scan.sbatch:100](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100) [bench_voronoi_partition_methods.py:217](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:217) [sharded_dynamics.py:1345](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1345)  
  But the repository contains no recorded census provenance beyond that docstring, and the batch does not check values. It is not a validation until it parses and rejects mismatches. Also do not compare a raw-target-64 score to timing legs deliberately reordered for 128; the production bench supports and warns about that distinction. [sharded_dynamics.py:1283](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1283) [mpas_s8_lloyd0_ladder.sbatch:50](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50)
tokens used
174,717
## Findings

- **BLOCKER: “validation” is only a comment.** The sbatch never compares JSON results with the census, never requires all three methods, and continues after a failed arm. `run_arm` logs `$?`, then returns the succeeding `date` status; the script has no `set -e` and proceeds to s10 regardless. A missing `pymetis` is explicitly tolerated, too. [mpas_schedule_cost_scan.sbatch:76](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:76) [mpas_schedule_cost_scan.sbatch:94](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:94) [mpas_schedule_cost_scan.sbatch:97](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:97) [mpas_schedule_cost_scan.sbatch:108](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:108)  
  As written, arms 1–2 do not validate anything and cannot gate quoting s10.

- **Your `gap <= 1 ⇒ ownership is the only path` conclusion is false.** `gap == 1` is inconclusive: a better colouring may remove one round (Class 1), or it may already be optimal (Class 2). The test correctly says this, but the batch’s decision rule still says it “REFUTES” recolouring and makes ownership the only path. [test_bench_voronoi_partition_methods.py:194](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:194) [mpas_schedule_cost_scan.sbatch:39](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:39)  
  Correct discriminator: `gap == 0` rules out a better **undirected edge-colouring of this graph**; `gap == 1` does not.

- **`coloring_headroom_rounds` is documented and printed backwards.** For `n_rounds=14, Δ=10`, Vizing guarantees a reduction of **at least** 3 rounds, while 4 remain possible. `max(0, gap-1)` is a lower bound on improvement, not the “MAXIMUM” improvement; the `<= 3` output is wrong. The test repeats the same error (“at most”). [bench_voronoi_partition_methods.py:194](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:194) [bench_voronoi_partition_methods.py:225](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:225) [bench_voronoi_partition_methods.py:317](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:317) [test_bench_voronoi_partition_methods.py:178](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:178)

- **(a)/(b): the bound is valid, but only for the implementation’s artificially symmetric graph.** `max_degree` is computed from the exact `comm_pairs` later coloured, and properness is asserted, so `gap == 0` genuinely proves optimality for that simple undirected graph. It is neither a cell-degree nor edge-degree mixup. [sharded_dynamics.py:1733](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1733) [sharded_dynamics.py:1815](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1815) [sharded_dynamics.py:1851](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1851) [sharded_dynamics.py:1864](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1864)  
  But `comm_pairs` is formed when either direction has a halo dependency, then every pair is forced into *both* ppermute directions—even when one send map is empty. [sharded_dynamics.py:1795](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1795) [sharded_dynamics.py:1923](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1923) [sharded_dynamics.py:1945](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1945)  
  Therefore Δ is not a lower bound for a redesigned *directed* schedule/packing that exploits one-way exchanges. “Only a new ownership objective remains” overclaims even at gap zero.

- **The provenance claim is false at row level.** `--lloyd` does reach mesh construction, and schedule scoring does not consume MPI `--halo-depth`; those two code paths are correct. [bench_voronoi_partition_methods.py:217](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:217) [bench_voronoi_partition_methods.py:279](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:279) [sharded_dynamics.py:1250](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1250)  
  But `lloyd_iterations` exists only once in enclosing metadata, not in each row/schedule block as the sbatch claims. Worse, the wrapper drops scorer provenance including `reorder_target` and `already_reordered`. A flattened/copied row cannot establish that it was raw, `lloyd=0`, or partitioned for the reported device count. [bench_voronoi_partition_methods.py:219](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:219) [bench_voronoi_partition_methods.py:345](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:345) [sharded_dynamics.py:1375](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1375)

- **The `--lloyd` test is vacuous.** It asserts the default value `50`; removing `lloyd_iterations=args.lloyd` would still pass. There is no `--lloyd 0` invocation or mesh-builder spy. [test_bench_voronoi_partition_methods.py:267](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_bench_voronoi_partition_methods.py:267)

- **(c): yes, the small even-device `rounds == n_dev - 1, gap == 0` cases are complete-graph saturation.** Δ then equals `n_dev - 1`, which is only possible for \(K_{n_dev}\). s6@16 escapes that regime because 13/10/10 is below 15; geometric is still very dense, but it is not complete-graph saturated. Also, “2560 cells/device” is wrong for the actual score: L6 has 40,962 cells and padding to 16 produces 40,976, hence **2561**. It remains safely above the 2,000 ppermute threshold. [voronoi.py:1259](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/voronoi.py:1259) [voronoi_partition.py:1211](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_partition.py:1211) [sharded_dynamics.py:1393](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1393)

- **(d): the invocation does statically match the docstring’s stated raw-mesh/default-target setup**—`--lloyd 0`, explicit method, then `spmd_schedule_cost(raw_mesh, n_dev, method=...)`, whose default target is `n_dev`. [mpas_schedule_cost_scan.sbatch:100](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:100) [bench_voronoi_partition_methods.py:217](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_voronoi_partition_methods.py:217) [sharded_dynamics.py:1345](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1345)  
  But the repository contains no recorded census provenance beyond that docstring, and the batch does not check values. It is not a validation until it parses and rejects mismatches. Also do not compare a raw-target-64 score to timing legs deliberately reordered for 128; the production bench supports and warns about that distinction. [sharded_dynamics.py:1283](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1283) [mpas_s8_lloyd0_ladder.sbatch:50](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50)
