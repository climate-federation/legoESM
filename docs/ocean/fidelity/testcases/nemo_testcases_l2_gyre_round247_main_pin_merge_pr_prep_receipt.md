# Round 247 — pinned-main merge and VORTEX-SMT PR preparation

**Status: IN PROGRESS.** This receipt is completed only after every frozen
trajectory and test gate below has run on the committed merge tree.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round247_main_pin_merge_pr_prep.md`.
Lane base: `03d9cd620`. Pinned main: `471bee222` (`origin/main-pin`). Merge
base: `cffa2ab79`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round247/`.

## 1. Merge construction and file audit

The preregistration was committed before the merge. The lane was 142 commits
ahead of the merge base and pinned main was 232 commits ahead. Git's
three-way merge completed with zero textual conflicts. The merge commit is
`cd1dcf7225423e3d60720de7c3b50966edcd65a5`; its initial tree exactly equals
the tree emitted independently by `git merge-tree --write-tree`. A trailing
blank line introduced by main in `tests/ocean/unit/test_omip2_applicator.py`
was then removed without changing executable code or an assertion; the
cleanup is `acf353f42` and `git diff --check` is clean.

The lane/main intersection under `packages/ocean` and `src/legoesm` is empty.
The ten main-side files named in Decision 98 were read in the merged tree:

| file | pinned-main change | disposition in the union |
|---|---|---|
| `ocean/coupler/__init__.py` | stop exporting the deleted legacy OMIP-2 applicator | intended; no remaining caller imports it |
| `ocean/coupler/omip2_applicator.py` | remove the defective out-of-step forcing applicator, retain in-step builders | intended; certified cards already force in-step |
| `ocean/dynamics/_future/__init__.py` | document parked components | documentation only |
| `ocean/dynamics/_future/gm_bvp.py` | park the test-only GM BVP implementation | intended move; no production caller |
| `ocean/dynamics/barotropic_implicit_mpas.py` | resolve PCG defaults by backend | MPAS-only; idealised lat-lon cards do not call it |
| `ocean/dynamics/ocean_model_mpas.py` | resolve the same backend bundle in the MPAS model | MPAS-only |
| `ocean/mpas_config.py` | make CPU/GPU PCG default bundles explicit | MPAS-only |
| `ocean/simple_ocean.py` | use a common MOST air-temperature reference | slab/simple-ocean only; not the 3-D C-grid route |
| `ocean/timestep.py` | delete a test-only timestep helper | no remaining reference |
| `src/legoesm/taxonomy.py` | delete retired taxonomy metadata | no remaining reference |

The scientific reachability audit also found a main-side change outside that
ten-file list which is relevant to this campaign: the shared Thomas solve in
`packages/core/legoesm/timestepping/tridiagonal.py` changed from indexed
buffer updates to `lax.scan` and gained an `n == 1` arm. The GYRE year and all
four SMT 100-day runs below exercise the merged solver directly, so this is
measured rather than inferred inert.

Evidence: `main_changed_files.txt`, `lane_changed_files.txt`,
`overlap_files.txt`, `main_ocean_source_diff.patch`,
`main_ocean_source_commits.txt`, `audited_main_files.txt`, and
`merge_tree.txt`.

## 2. Certified scientific results

### GYRE

PLACEHOLDER_GYRE_RESULTS

### VORTEX-SMT mini-ladder

PLACEHOLDER_SMT_RESULTS

### Control cards and DINO

PLACEHOLDER_CONTROL_RESULTS

The merge therefore either satisfies or refutes the preregistered zero-move
claim mechanically; no certified pin is silently re-baselined.

## 3. Tests, citations, and independent review

PLACEHOLDER_TEST_RESULTS

PLACEHOLDER_REVIEW_RESULTS

## 4. Verdict and OPEN

PLACEHOLDER_VERDICT

OPEN: the operator may open the PR from the merged lane using
`docs/ocean/fidelity/testcases/nemo_testcases_l1_vortex_smt_pr_summary.md`.
No physics walk, configuration choice, record acquisition, push, or PR action
is authorised in this round.

