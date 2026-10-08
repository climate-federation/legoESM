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

The fresh full ladder compares all 954 certified rows with zero aggregate or
cell movement, `max_worsening_ulps=0`, no status change, and first-over-bar
kt=3 on T/S/u/v/ssh in both arms. The shorter 70-row trajectory view also has
zero movement.

A fresh 360-day member was run on CPU. All 40 reported field/day scores equal
round 237 exactly. The eight T RMS values are:

| day | wet T RMS vs NEMO [K] |
|---:|---:|
| 30 | `2.3432419318363155e-06` |
| 60 | `1.4793244000905481e-05` |
| 90 | `1.6332666348876788e-05` |
| 120 | `1.0965903952280960e-04` |
| 180 | `6.1153308005679761e-05` |
| 240 | `6.5816987106668941e-05` |
| 300 | `5.4660367722690732e-05` |
| 360 | `5.4077212586815052e-05` |

The day-30/day-240/day-360 snapshot SHA-256 values are respectively
`3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`,
`2e2b895c72b3dd0218f22a06078d944cbf92c91f75493c7d4e21ddc7eb985abe`,
and `5af258eff135981fa80bfb3d4c354f034f66fcde9d9c3712f1608aff88e37646`,
exactly the round-237 pins. The first score invocation pointed at the daily
record root and refused because it could not find the 60-day restart; the
corrected invocation used the admitted `year_fromrest` record and is the only
one cited. No run was repeated and no result from the failed invocation was
used.

### VORTEX-SMT mini-ladder

Each card's scorer first reran its ten-step trajectory and reported
`REPRODUCED`. The four 50-row registries therefore have zero movement. The
fresh daily results are:

| card | day-100 wet T RMS vs NEMO [K] | exact comparison to certified run |
|---|---:|---|
| SMT-1 | `4.3321114781972461e-05` | 800/800 daily scalars equal |
| SMT-2 | `8.1037591477894766e-06` | 800/800 daily scalars equal |
| SMT-3 | `1.7729713625071864e-04` | 800/800 daily scalars equal |
| SMT-4 | `2.5527080520554426e-04` | 800/800 daily scalars equal |

Thus the combined comparison covers 3,200 T/u/v/ssh daily RMS and maximum
values with zero changes, not just the four requested endpoints. The initial
sequential replay was deliberately interrupted after SMT-1 day 59 so the four
complete clean-tree arms could run in parallel; its partial directory is not
used as evidence. The complete roots and logs are `smt1_100day` through
`smt4_100day`, and `smt_full_compare.txt` is the exact comparator.

### Control cards and DINO

The six flat VORTEX cards at 30/15/10 km, both base seamount cards,
LOCK_EXCHANGE, and OVERFLOW each compare 50 rows against the round-237
references. All ten comparisons pass with zero improved cells, zero worsened
cells, `max_worsening_ulps=0`, no status change, and unchanged first-over-bar.

PLACEHOLDER_DINO_RESULT

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
