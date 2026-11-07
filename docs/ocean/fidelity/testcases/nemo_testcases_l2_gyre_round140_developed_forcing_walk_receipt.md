# NEMO testcase L2 GYRE phase 3 — round 140 developed forcing walk receipt

Date: 2026-09-21

Incoming lane tip: `63ef82a7f7f5d6519df2cb4f5a6a6e1d5c8cecae`

Preregistration commit: `d99ef4453`

Accepted measurement/retraction commit: `089db8c4d`

Final acquisition-contract commit: `fb4e4ea03`

Status: **STOPPED_FOR_RECORD — the admitted developed-state walk proves that
the first non-bit boundary is the completed incoming U slow forcing, with the
V boundary next. The mismatch is already present before the initializing
Coriolis subtraction. The required upstream three-dimensional-RHS record was
preregistered, implemented, syntax-proved, and planted, but its run refused
with exit 64 before `makenemo` because the NEMO configuration tree is not
writable in this sandbox. No physics, configuration, carried state, or
certified trajectory changed.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round140/`

## Outcome first

At NEMO step 1081 (day 180, first step), the production-JIT model's incoming
U slow forcing differs from NEMO on all 580 wet U faces, with maximum absolute
difference `4.2854247978022983e-13 m s-2`; incoming V differs on all 570 wet V
faces, with maximum `4.433308633699682e-13 m s-2`. These are the first
non-bit rows in compiled order. They already account for the final forcing's
`4.2854247978022983e-13` U and `4.4333086294645174e-13` V maxima. The
initializing Coriolis residual is nine orders smaller: at most
`6.485096002415737e-22 m s-2` in U and
`7.940933880509066e-22 m s-2` in V.

This names the completed incoming `Ue_rhs` boundary as the magnitude owner of
the observed frozen-forcing gap; it does **not** yet name an HPG, LDF, VOR,
WZV, KEG, ZAD, depth-average, drag, or wind statement. The producing record is
the missing discriminator. The next compiled statement therefore remains
upstream in `stp2d`; no later Coriolis or external-mode statement owns this
walk.

## Compiled-source citations

The exact compiled Round-139 program first copies `Ue_rhs`/`Ve_rhs`, calls
`dyn_cor_2D`, records the incoming and returned operands, subtracts the
Coriolis terms through `ssumask`/`ssvmask`, and records the results at
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:292-347`.
The called four-point U and V formulas, including their signs and written
association, are
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:1359-1382`.

Upstream, the exact compiled program accumulates HPG, LDF, VOR, WZV, KEG, and
ZAD into `Krhs`, writes the established three-dimensional slow-forcing field
list, depth-averages it, then applies drag and wind at
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:139-228`.
It passes that completed pair to the external solver at
`GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:297-298`.
The new card adds only a step-1081 duplicate of the already compiled writer
around those boundaries; it removes or reorders no executing statement.

## P0 — record readmission

The acquired Round-139 stream was readmitted before every accepted
comparison. Its exact contract is:

- record: `oracle_slow_forcing_split_kt00001081.bin`;
- size: `33,852` bytes;
- SHA-256: `0fee139d96d9a3731ad1d40a6dc28ead3b95b4a1bbb68f51d7cc99baf1215e95`;
- producer: `63ef82a7f7f5d6519df2cb4f5a6a6e1d5c8cecae`;
- header: fp64, step 1081, `Kmm=1`, owned extent 32 by 22, six fields;
- source replay: BIT for U and V; and
- closed manifest: 19 members, inherited restart/process/external/QCO hashes
  unchanged, `STOP 0`, `RUN_DONE`, and all stamp/header/truncation/replay-ULP/
  passive-admission markers present.

The step-1080 bridge mapped T, S, u, v, ssh, all six barotropic histories,
TKE, avm, avt, dissl, and the surface avm field with zero unequal cells. No
post-entry model field was silently substituted.

## P1 — direct developed-state rows

The authoritative table is production JIT through `self._step_jitted` on CPU,
fp64/x64/libm. The eager column is a control, not the production verdict.
Rows are in compiled order; masks are scored over their complete owned arrays.

| boundary | JIT unequal / scored | JIT max abs | eager unequal / scored | eager max abs | disposition |
|---|---:|---:|---:|---:|---|
| incoming U | 580 / 580 | `4.2854247978022983e-13` | 580 / 580 | `4.2854247978022983e-13` | first non-bit; magnitude owner boundary |
| incoming V | 570 / 570 | `4.433308633699682e-13` | 570 / 570 | `4.433308633699682e-13` | next non-bit boundary |
| Coriolis U | 515 / 580 | `6.485096002415737e-22` | 520 / 580 | `6.485096002415737e-22` | non-bit last-order residue; downstream |
| Coriolis V | 502 / 570 | `7.940933880509066e-22` | 494 / 570 | `7.940933880509066e-22` | non-bit last-order residue; downstream |
| U mask | 0 / 704 | `0` | 0 / 704 | `0` | BIT |
| V mask | 0 / 704 | `0` | 0 / 704 | `0` | BIT |
| final U | 580 / 580 | `4.2854247978022983e-13` | 580 / 580 | `4.2854247978022983e-13` | reproduces Round 138 |
| final V | 570 / 570 | `4.4333086294645174e-13` | 570 / 570 | `4.4333086294645174e-13` | reproduces Round 138 |

The write-only callback's returned state is BIT against an independently
compiled ordinary step across 198,956 cells and 23 leaves. Its recorded final
U/V fields are also BIT against the actual external-call fields on all
580/570 wet faces. Thus the accepted observer is passive at both required
boundaries.

Replacing only the incoming pair by NEMO's recorded pair makes the incoming
rows BIT and leaves both Coriolis rows unchanged, but does not close the final
result bit-for-bit: U retains 268 unequal faces at
`6.88214269644119e-22`, and V retains 247 unequal faces at
`8.470329472543003e-22` under production JIT. Eager retains 268 U and 246 V
faces at the same maxima. This directed arm removes essentially all of the
`1e-13` magnitude, while proving that incoming forcing is not the sole
last-bit owner.

## Frozen predictions and falsifiers

| frozen prediction | observed result | disposition |
|---|---|---|
| incoming U is first and differs on 580 / 580 wet faces | 580 / 580, max `4.2854247978022983e-13` | CONFIRMED |
| incoming V is next and differs on 570 / 570 wet faces | 570 / 570, max `4.433308633699682e-13` | CONFIRMED |
| both masks are BIT | 0 / 704 unequal for each | CONFIRMED |
| both initializing Coriolis fields are BIT | 515 U and 502 V JIT cells unequal | REFUTED |
| NEMO incoming pair makes final pair BIT | 268 U and 247 V JIT cells remain unequal | REFUTED |
| ordinary step reproduces Round 138 and observer moves no returned cell | exact final census; 0 / 198,956 returned cells move | CONFIRMED |
| upstream geometry/masks/reference reciprocal are BIT and completed `Krhs` U is first | required record could not be acquired | UNMEASURED |

The two refutations are retained. Neither overturns the magnitude ordering:
their residual is approximately nine orders below the incoming and final
forcing maxima.

## Instrument retractions and plants

Two observer designs were rejected rather than promoted:

1. The initial return-value `expose_live_stage_operands` arm reproduced the
   returned model state but perturbed the internal external-call forcing by
   159 U and 150 V faces, maximum `1.6940658945086007e-21`. It is not used by
   the accepted measurement.
2. Expanding the write-only callback to materialize 19 additional upstream
   fields caused the full-step returned-state identity check to fail at commit
   `3c0609f13`. Commit `089db8c4d` retracts that mode: the production callback
   again exposes only the eight passivity-proven fields, and the unavailable
   developed-RHS comparison now raises a named `RETRACTED` failure rather than
   printing a result.

The accepted registry-omission plant exited 1 with
`ROUND140 DEVELOPED DEVELOPED-MISSING-ROW STATUS PLANT-FIRED`. The consumed
incoming-U one-ULP plant crossed the production subtraction and exited 1 with
`ROUND140 DEVELOPED INCOMING-ULP STATUS PLANT-FIRED`. All inherited Round-139
record controls were rechecked through P0.

## P2 — conditional upstream acquisition

Because P1 confirms the incoming boundary, the conditional acquisition was
activated. The new target is `GYRE_OMIP_L2_P3_SM_R140RHS`; its exact record
contract is the existing six 36-by-26-by-31 three-dimensional fields plus the
depth-mean, reference reciprocal, post-drag, drag-coefficient, density,
stress, live reciprocal, and post-wind fields: `1,486,548` bytes. The reader
requires magic `NEMO_L2_R140RHS`, version 3, step 1081, `Kbb=1`, `Krhs=3`,
fp64 dimensions and field sizes, exact EOF, finite values, exact stamp and
producer, a closed ordered manifest, inherited-byte passivity, and literal
depth/wind replay.

A final adversarial audit caught and corrected a would-be replay defect before
handoff: after the parser removes NEMO's single non-contributing `jpk` slot,
the older 31-slot helper would also have skipped the deepest retained physical
level. The Round-140 replay now sums all 30 retained levels, and a non-vacuous
test places the only nonzero product at level 30. The closed manifest also
includes the parent external-record validation, and every inherited stream
stamp is checked against its producer and digest.

The additive patch has zero removed lines. Dry preprocessing and
`gfortran -fsyntax-only` printed `SYNTAX_PROOF_PASS stp2d.f90`; its layout
plant removed the post-drag write and exited 69 with
`STATUS PLANT-FIRED: layout`. The real run then refused before target creation:

```text
REFUSE: NEMO configuration directory is not writable
EXIT=64
```

The target configuration and target run directory remain absent. Therefore
the conditional `Krhs` prediction is UNMEASURED, the expanded upstream
observer is not a candidate, and no upstream statement is inferred.

## Review, citation gate, and focused tests

The required separate read-only Codex review result is recorded verbatim
below after the final diff review:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
EXIT=1
```

Independent review was unavailable in-sandbox and emitted no `SHIP`, `HOLD`,
or `DO NOT SHIP` verdict. The standing operator rule permits continuation in
this case. Nothing scientific or production-facing is being landed; the
result is a record stop.

Citation-gate and shifted-citation-plant results:

```text
clean: PASS; 4 citations; 0 unmapped; 0 failures; 0 failing map entries
shifted stp2d citation plant: FAIL as intended; exit 1
```

The first clean attempt refused the post-wind anchor because its prefix occurs
at three writer boundaries. The final map pins occurrence 3 at the actual
line-228 endpoint; it does not weaken the anchor or widen the cited range.

Focused CPU/x64 test summaries:

```text
82 passed in 2.58s
```

The focused set covers the private live-operand hook contract, the admitted
external-step parser, the external-step walk, the extended slow-forcing
reader/registry/replay controls, every citation-gate unit control, and the
full fidelity time-level unit file.

## Artifacts

| artifact | SHA-256 |
|---|---|
| `developed_split_jit_final.json` | `04eb8c18c3ff9bae2b96e6884de43fcc670b81666d678695e15ea84a21d45a67` |
| `developed_split_eager_final.json` | `fb11dbcf58073e2e46363b35a34b5979d96a6fc855dfbfda15bc4f56dae33a63` |
| `missing-row_plant_final.log` | `8309b540fe6e1ba08dbbf2b17bfef90b7cfe9d677cc4c78f72c331e504f8c099` |
| `incoming-ulp_plant_final.log` | `fa9f63b2019f17ce5cbdf7be1c60abc61ff9b3fab5a8eef1b5735d404ac1e4bc` |
| `rhs_preflight.log` | `8b2f596fad6f9f21cdb5b52efdccd2b6cff2052dfc84e761dbd2453779487956` |
| `rhs_layout_plant.log` | `6b45a1ab9bd6ba63f2b5552bb24c3956092c6dec24ac9c9a762ba0054c91581d` |
| `rhs_acquisition_refusal.log` | `e858fc1137e60ec08c846e076595f8ab32e1ecba09a7be5e920fd36b88307c88` |
| `citation_gate_final.json` | `5e3c4f6586fde0348227711ff4da21e354ad6264659525cdc600036eeb981288` |
| `citation_gate_shifted_plant_final.json` | `a9effd55b20c455c9a92279e930c59a9c0e22325e28809fe6041e6d12a0952d5` |
| `codex_review.log` | `3a9581859fa4443d6ea3767dcf85865a0c9734478edbe95f4a3fc1d59642cb16` |
| `focused_tests.log` | `a5fb1f4b5f9128fc2cb52faae3a78ba0c9d3bf0bb0bdc7abe1b5ce130fb1ca27` |
| `focused_tests_final.log` | `4fc84ac5a40d2b9f1a78407842b5d6377e155ed28734c012e93fafbbaa2482d4` |
| `stp2d_round140.patch` | `d26262941a229b168e22c87c631f25fa7d2b65d61ad1426b376629ce64629a75` |
| `run.sh` | `e7906557524578a923876cddd7ff82fb640f2ceca45e914f30231aae2dfc9189` |

## Decision-43/45 gate and campaign surfaces

There is no candidate physics change, so there are no moved certified rows
and no landing verdict. The ladder, month, and year are not rerun. Their
immutable headline values remain:

| row | unchanged value |
|---|---:|
| kt2 U | `2.7377110452773967e-12 m/s` |
| kt2 V | `3.2849219221489645e-12 m/s` |
| kt3 T | `8.659373840202989e-7 K` |
| kt3 S | `7.027291104577671e-8 g/kg` |
| day-30 T3D RMS | `6.890431487825909e-5 K` |
| day-240 T3D RMS | `1.644674193e-2 K` |
| day-360 T3D RMS | `1.122357391e-2 K` |

DINO, LOCK_EXCHANGE, OVERFLOW, and the tank cards execute no changed
production physics and retain their incoming certified results. ORCA2 remains
`UNMEASURED-WITH-SPEC`: the same developed native forcing registry and a
card-specific admitted entry are required before an identity claim.

## OPEN — round 141

1. Run the committed acquisition script in an environment where the NEMO
   configuration tree is writable. Require
   `ROUND140_DEVELOPED_RHS_READY`; any refusal remains a record stop.
2. Admit the exact 1,486,548-byte record, its producer/stamp/manifest, literal
   depth and wind replays, inherited restart/process/external/QCO/Round-139
   byte identities, and all header/truncation/replay-ULP/stamp/passivity
   plants. Cite the new target's compiled branch before using its numbers.
3. Do not restore either rejected observer. Prove a production-JIT upstream
   observation path passive against an independently compiled ordinary step
   before scoring any three-dimensional field.
4. Then adjudicate the frozen conditional prediction in compiled order:
   thickness U/V, completed `Krhs` U/V, masks U/V, consumed reciprocal U/V,
   and depth mean U/V. If `Krhs` U is first, request the minimum same-step
   cumulative HPG/LDF/VOR/KEG/ZAD split; do not infer its owner.
5. No downstream bit walk, held rest-state patch, configuration change, or
   trajectory run substitutes for the missing direct record.

`ACQUISITION_NEEDED`:
`/tmp/autopilot-work-27285181/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round140_rhs/run.sh`

`DECISION_NEEDED`: `NONE`

`ROUND_STATUS`: `STOPPED_FOR_RECORD`
