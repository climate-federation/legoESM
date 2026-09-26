# PR #1695 / #1696 rebase resolution receipts

Date: 2026-08-30  
Session: `01a04e34-d1fb-73e0-b25a-177641f0a246`

This is the review-before-force-push record for rebasing the two DINO fidelity
branches after #1682/#1686/#1687/#1688/#1689 entered main.  No branch was
pushed, and no GPU or `mpirun` command was used.

## Rebase identities

The target is the actual fetched `origin/main` history, not the stale local
`main` in the shadow clone:

- target main: `b794c0618e287ebf1d364a8713c3c504ac2eb01c`
- old common ancestor: `782b0d7887277c88bcaa9c1be24eedad5447d9ae`
- #1695 old tip: `f862e1554911f0901ab15b7ede80de7a74d2f1ac`
- #1695 rebased code/test tip before this receipt:
  `e0f167b18fd005917efbc991d2b5c6784e0998e6`
- #1696 old tip including the round-78 pin correction:
  `9496a66b8c756ec75a1723febbde8e7cc61912f2`
- #1696 rebased code/test tip before this receipt:
  `2836a05f596fc0bfb43bd58034438d4b3d7d4142`

#1695 replayed 72 commits, then added one review correction.  Range-diff
classifies 67 patch-equivalent, five context-adjusted, and one new commit.
#1696 replayed 525 of 527 commits, then added three
post-rebase contract/review/packaging commits.  Range-diff classifies 518
patch-equivalent, seven context-adjusted, two dropped-as-upstream, and two new
code/test commits plus one documentation-only pin commit.  The two commits
Git recognized upstream and skipped are:

- `4ff604100fe4e5cd8596b8d778ace1b78d665193`, NEMO-latitude V-face width;
- `9445d8228c8e89baa550f8909381f3ef04b32261`, CPU/GPU bit-match scope.

## #1695 conflict receipts

### 1. End-wall executable handoff

- old/new commit: `4869209d9188` -> `439213645972`
- file: `scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py`
- conflict: branch per-step `eta_step` payload assembly overlapped main's fp64
  snapshot `_stored_days` and always-on `_red_kwargs` assembly.
- resolution: retain the per-step eta replacement and its time/dt stamps, then
  iterate main's single `_stored_days` set and append `_red_kwargs`.  The old
  branch-local `snaps` write loop was discarded because it would duplicate the
  resolved-day contract.

### 2. Basin re-verdict scorer

- old/new commit: `d6dc89e91c9a` -> `507921f1f71b`
- files: `kamm_twin_90d.py`, `tests/ocean/unit/test_dino_1226_instruments.py`
- run-config hunk: retain the branch's barotropic perturbation/daily transport
  provenance keys while retaining main's rule that storage-only `fp64_3d` is
  excluded from the scientific run-config identity.
- save hunk: compute main's snapshot/reduced-series stamps, then run the
  branch's entry/exit producer-SHA and tracked-dirty identity gate before any
  artifact is saved.
- test hunk: retain all main fp64 snapshot/reducer tests and the public
  `restart_elapsed_seconds` assertion, then append all branch basin-scorer
  controls.  The obsolete private-helper assertion was discarded.

### 3. Bridge-Omega discriminator

- old/new commit: `9e339ad1b203` -> `1e3b5618fc1b`
- file: `kamm_twin_90d.py`
- conflict: main added `fp64_3d` while the branch added `bridge_omega` at the
  same `run_twin` and CLI call signatures.
- resolution: retain both independent parameters and forward both CLI values.

## #1696 conflict receipts

### 1. Literal Langmuir transcription

- old/new commit: `5b13cfbdee3e` -> `7dc88298e338`
- files: `tests/ocean/unit/test_dino_1226_instruments.py`,
  `tests/ocean/unit/test_tke_nemo_terms.py`
- instrument-test hunk: retain main's public clock assertion and complete fp64
  snapshot/reducer suite, then append the branch's literal-vs-vectorized
  Langmuir CLI test.
- TKE-import hunk: retain main's backward-Euler and surface-Dirichlet helpers
  and the branch's `TKEEntryN2Bundle`; all are used later in the merged test.

### 2. Metric climate treatment arms

- old/new commit: `f1edc329f668` -> `1779149715db`
- file: `kamm_twin_90d.py`
- post-run hunk: retain main's `_stored_days`, storage-dtype and reduced-series
  receipts, then enforce the branch's producer entry/exit SHA and dirty-state
  identity before saving.
- payload hunk: retain branch per-step eta/timing output first, write main's
  resolved fp64/fp32 3-D snapshot days once, then add main's reduced series.
  The duplicate branch `for d in snaps` writer was discarded.

### 3. Climate snapshot contracts

- old/new commit: `40b2e9644d8f` -> `3c5f54558806`
- file: `kamm_twin_90d.py`
- clock/snapshot hunk: use main's public `restart_elapsed_seconds`, resolve the
  snapshot day set once before stamping `run_config`, and reuse it at capture.
- dtype hunk: retain main's `snapshot_dtype(fp64_3d)` and reduced-series setup
  around that single resolved day set.

## Post-rebase contract corrections

Commit `e0f167b18fd0` on #1695 fixes the independent review's live-path finding:
main made `restart_elapsed_seconds` public, but two corrected-stress paths still
called the removed private spelling.  Both live calls now use the public helper;
a compatibility alias remains only for frozen end-wall/channel probes, and a
red-capable test requires the two public live calls and alias identity.

Commit `650b73e6b774` is deliberately separate from replayed physics.  It makes
five narrow contracts explicit:

1. Provide `_restart_elapsed_seconds = restart_elapsed_seconds` as a temporary
   compatibility alias for frozen branch scorers while all live harness calls
   use main's public spelling.
2. Advance the exact `BarotropicConfig` nesting ratchet from 36 to 37 for the
   literal time-mean transport-accumulation selector.
3. Pin a synthetic surface-tendency-placement test to generic ZAD/WZV; the test
   has no restart operands and is not a literal-QCO integration test.
4. Initially align the realized `avt`/`avm` time-level assertion to the live
   `zdfphy.F90:311-344` closure-copy/EVD provenance.  Reviewer 1 then showed
   that this exposed, but did not resolve, duplicate registry literals; the
   following review commit supersedes this part of the correction.
5. Run the raw EEN bridge-operand identity assertion under an explicit fp64
   precision policy, restoring the prior policy afterward.  This prevents an
   fp32 storage cast from masquerading as a reconstruction error.

Items 3 and 5 also fail at the old #1696 tip, proving they were pre-existing
test-isolation defects rather than rebase-created physics changes.  Items 1,
2, and 4 are main/branch contract drift exposed by the rebase.

Commit `7723918c2b2f` on #1696 resolves the independent review's dump-registry
finding.  It removes the later duplicate `dump_avt.bin`/`dump_avm.bin` dict
literals, keeps one entry per stream, and combines main's actual writer sites
(`MY_SRC ldftra.F90:955-956`) with the branch's closure/EVD and `lbc_lnk`
composition provenance (`zdfphy.F90:311-344`).  A source-level uniqueness test
is red on the duplicate-literal form that Python otherwise accepts silently.

Commit `2836a05f596f` on #1696 resolves reviewer 2's round-78 packaging finding.
The pre-rebase producer is unreachable from the rebased ref, so all four
handoff pins now name reviewed code/test tip
`7723918c2b2ff42071d937988574dfccc5fcf197`.  At packaging tip `2836a05f596f`,
`git merge-base --is-ancestor $producer HEAD` passes and the handoff's exact
tracked model gate reports no paths from `packages/core`, `packages/ocean`, or
`src` changed after the producer.

## Validation receipts

All commands set checkout-first `PYTHONPATH`, `JAX_PLATFORMS=cpu`,
`CUDA_VISIBLE_DEVICES=''`, and `JAX_ENABLE_X64=1`.

### #1695

- `test_dino_1226_instruments.py` + `test_nemo_state_bridge.py` after the
  review correction: **138 passed**, one expected precision warning, 8.04 s.
- `tcarry_basin_reverdict.py --self-test`: exit 0; every selector/stagger,
  dtype, `rn_Uv`, unregistered-perturbation, classifier and receipt plant
  passed through real temporary NPZ files.

### #1696

- barotropic/geometry tranche: **240 passed, one skipped** after the sole
  stale 36-field ratchet was corrected and rechecked in the final tranche.
- TKE/Redi/ZDF tranche: **383 passed**, 302.39 s.
- final harness/config/state/CLI tranche from a clean tracked tree:
  **401 passed**, 11 warnings, 231.18 s.
- after the review registry consolidation, the entire time-level registry
  suite: **24 passed**, 0.35 s.
- `zdf_wall_epoch_tke_core_score.py --self-test`: exit 0; legal-arm,
  decomposition, component classifier and disposition controls passed.

### Historical round-77 scorer qualification

An offline replay against every retained round-54..76 and Redi operand artifact
reproduced all scientific rows 8.3--8.10 and all scientific metric tables
exactly.  Historical artifact SHA256 is
`dcc0cff4c63b30024794ad25b023b65223fe87137e5052b6d7dc478066973d14`;
the rebase-check artifact SHA256 is
`e5b5d0f83ed582471c5f6e65d6da44bfa2ebc04307cc20aaf6ec117493cfc7be`.
It is not a byte-for-byte disposition replay: the current round-78 scorer hash
is `4498395b...` rather than round 77's `b284091f...`, adds sign/zonal-roll
plants, and marks the retained round-77 invocation `INVALID` because its new
8.4 zonal-roll plant is inert.  This is recorded as scorer-epoch drift, not as
a reproduced historical verdict and not as a rebase physics change.

## Independent review

- Reviewer 1 found a **HIGH** #1695 runtime defect: two default
  corrected-stress paths called the removed private clock helper.  Accepted and
  fixed by `e0f167b18fd0`; the full 138-test scoped suite passes afterward.
- Reviewer 1 found a **MEDIUM** #1696 provenance defect: duplicate realized
  `avt`/`avm` keys silently discarded main's writer-site citations.  Accepted
  and fixed by `7723918c2b2f`; the full 24-test registry suite passes afterward.
- Reviewer 2 found a **HIGH** #1696 packaging defect: round 78 still pinned the
  unreachable pre-rebase producer, and its model-diff gate would stop.  Accepted
  and fixed by `2836a05f596f`; ancestry and tracked model-diff admissions pass.
- Reviewer 1 final result: **PASS** at #1695 `e0f167b18fd0` and #1696
  `2836a05f596f`, including a second bounded check of the repin.  The producer
  occurs exactly four times, the stale SHA occurs zero times, the producer is
  the direct parent, and the tracked model diff is empty.
- Reviewer 2 final result: **PASS** at the same tips.  Independently verified
  the conflict/range adjustments, upstream-equivalent skips, red-capable test
  corrections, NEMO registry citations, exact scientific-table scorer replay
  qualification, and corrected producer admission.

Both independent reviews have no remaining findings.

No force-push is authorized by this document.
