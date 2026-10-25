# ORCA2 round 333 — first wet-value departure and tracer acquisition handoff

Date: 2026-10-25

Base: `072e4aa1b1c8`; preregistration: `fb7aa3cffc`; measurement gate:
`c238f7232c`. Disposition: **ACQUISITION_NEEDED**. This round changes no
model statement or configuration value, and Codex did not run NEMO.
Signed-zero-only dry-cell rows were excluded exactly as instructed; no scoring
rule was changed.

## First wet-value and first over-bar rows

The two requested boundaries coincide for both rung-10 labels. The metrics
below are measured independently for each label in campaign row order. RMS is
reported both over the complete canonical row and over NEMO's wet support so
that the denominator is explicit.

| label | boundary | first row | max abs | full-row RMS | wet RMS | unequal wet | maximum `(i,j,k,rank)` | location |
|---|---|---|---:|---:|---:|---:|---|---|
| given NEMO entry | first wet value | kt 2 stage 3 `T` | `7.678080393702658e-12 K` | `1.2687664596813964e-14 K` | `1.561494903752516e-14 K` | 18 | `(32,147,5,0)` | fold, under ice (`fr_i=0.9962616058070369`); not seam, coast, or open ocean |
| given NEMO entry | first over bar | kt 2 stage 3 `T` | `7.678080393702658e-12 K` | `1.2687664596813964e-14 K` | `1.561494903752516e-14 K` | 18 | `(32,147,5,0)` | fold, under ice (`fr_i=0.9962616058070369`); not seam, coast, or open ocean |
| independent | first wet value | kt 2 stage 3 `T` | `7.678080393702658e-12 K` | `1.2687664596813964e-14 K` | `1.561494903752516e-14 K` | 18 | `(32,147,5,0)` | fold, under ice (`fr_i=0.9962616058070369`); not seam, coast, or open ocean |
| independent | first over bar | kt 2 stage 3 `T` | `7.678080393702658e-12 K` | `1.2687664596813964e-14 K` | `1.561494903752516e-14 K` | 18 | `(32,147,5,0)` | fold, under ice (`fr_i=0.9962616058070369`); not seam, coast, or open ocean |

The unique maximum is not the first unequal word: the latter is global
`[j,i,k]=[147,31,2]`. All 18 unequal words are wet and value-unequal. The
normalized maximum is `2.5511626431537103e-13`, over the registered
`1e-15` bar, and the oracle normalization scale is
`30.0963970850997 K`.

Salinity departs at that same kt-2 stage-3 boundary on both independently
constructed labels: 13 wet words are unequal, with maximum absolute difference
`9.65e-11`. It follows T in the registered row order and therefore does not
change the first-wet-row identity above. Every earlier T and S row contains
signed-zero differences only, never a wet value difference.

The clean CPU/fp64/production-JIT artifacts were produced at
`c238f7232cc6cf24a7243cd6bc554c599cc6e0fa`:

- `orca2_rounds/round333/first_wet/given_nemo_entry.json`, SHA-256
  `b4c72b84aeaa22d58d29299c9b13af18ba403e31beae7bedd0007226f2bcd930`;
- `orca2_rounds/round333/first_wet/independent.json`, SHA-256
  `09eb4aa025907933847c4b4644dc0be301b96d89ec52ccb085dc3f4f3fea266f`.

## Offline source-order walk

The admitted round-5/326 trajectory establishes the row endpoints; the
round-328, round-331, and round-332 records refine vertical-mixing, TKE, and
kt-1 source-order boundaries. None records the inner kt-2 stage-3 tracer
boundaries. In the executing record build, stage 3 first zeroes `Krhs` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:604`, then
executes advection at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:637`, surface
forcing at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:645`, solar
penetration at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:717`, lateral
mixing at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:734`, the
enabled bottom/damping calls in
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:738-740`, and
the implicit tracer solve at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:749` before the
already recorded completed stage and fold exchange.

The earliest possible owner in that order is therefore the `tra_adv` call at
line 637. Its legoESM counterpart is the production stage-3 `_flux_pair`
call at `ocean_model_latlon_cgrid.py:2927-2928`. This assignment is
**PLAUSIBLE, not CONFIRMED**: the existing kt-2 records jump from the preceding
completed stage to the completed stage-3 frame, so they cannot distinguish
advection from any later tracer statement. No single statement is eligible
for a Decision-122 change.

## Acquisition extension

The committed operator-only extension imports the reviewed round-328 driver
and extends the reviewed round-332 writer rather than replacing their
controls. At kt 2 stage 3 it records raw, signed-zero-preserving `Krhs` on both
ranks after zeroing, advection, surface forcing, solar forcing, lateral mixing,
and the enabled bottom-heat-flux/BBL/damping group. It then records pre-fold
`Kbb`, `Kmm`, `Kaa`, and `r3t` after the implicit solve and after the inherited
kt-1 dump block. Every field has declared `jpi,jpj,jpk` (or `jpi,jpj`) extents
and a checksum. The existing stage frame supplies the post-boundary-condition
endpoint; admission rechecks the inherited record and compares rank-0 `Kmm`
to the admitted stage-2 state.

The corrected driver adds exact landing checks at the six cited source-order
locations plus the post-damping boundary, with one misplacement plant per
observer. It compiles and runs the extended writer for two ranks under
`-fcheck=bounds`, including raw `-0.0` cells, and has checksum, truncation,
time-level, wet-zero, and wet-Kmm plants. The arithmetic comparison strips all
r332/r333 observer calls from its baseline and reports its floating-point
instruction census. `--admit-existing` is pinned to the retained run's producer
commit and binary SHA; all `admit-r333-*.log` files remain inside retained
evidence and final admission runs inside the refused-output retention trap.

At committed tool revision `251029d2c`, the clean `--preflight-only` run
passed 77 inherited tests and 6 round-333 tests, all seven placement plants,
the three enabled-selector plants, both-rank bounds/sign fixtures, checksum and
truncation plants, deterministic passivity across 140 inherited files, and
eight real-Fortran syntax proofs. The observer-free versus instrumented object
comparison reported an unchanged census of 1,367 floating-point instructions;
the same-program proof reported 409 identical preprocessed units and 421
identical normalized objects. It printed
`ORCA2_ROUND333_TRACER_PREFLIGHT_READY` and did not build or execute NEMO.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r333/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_tracer_stage3_kt2_np2`.

## Preregistered verdicts and stop

- **R333-P1 — REFUTED in its RMS clause.** The predicted row, maximum, and 18
  wet unequal words are confirmed, but the wet-support RMS is
  `1.561494903752516e-14 K`, not the published full-row RMS
  `1.2687664596813964e-14 K`; equal dry words still change the denominator.
- **R333-P2 — CONFIRMED.** Both labels have identical row identities, metrics,
  maximum location, wet census, and flags.
- **R333-P3 — CONFIRMED.** The maximum is on global fold row 147, rank 0, off
  the cyclic seam; the measured additional flag is under ice, not coast or
  open ocean.
- **R333-P4 — PLAUSIBLE; ACQUISITION_NEEDED.** The first wet departure is in
  the stage-3 tracer program, but the admitted records skip its first
  discriminating statement.
- **R333-P5 — CONFIRMED as a stop condition.** There is no confirmed
  single-statement owner, so Decision 122 is not opened and no production
  edit, cross-card claim, DINO/month claim, or gradient claim is made.

The per-receipt citation gate passes 8 citations with zero failures, zero
unmapped citations, and zero map-audit failures; all nine self-test controls
fire. The exact default ORCA2 landing battery passes 152 tests in 1,133.45
seconds, including the required receipt-citation test. The explicit
private-import ratchet passes 6 tests in 5.54 seconds.

No configuration value changed: **ASKED**. The next admissible action is the
operator-only acquisition above, followed by admission and offline replay.

Independent reviews: **HOLD** (Claude and Argon) on the superseded acquisition
tool. Every enumerated finding is repaired above; operator rereview is still
required before acquisition.

## Resume 2: retained-run admission review repair

Claude and Argon confirmed every earlier acquisition repair and narrowed their
remaining HOLD to the two retained-run entry paths. Both are repaired without
changing an observer, a model statement, a configuration value, or a score.

The round-333 wrapper now adapts both the canonical target and the refused-run
basename predicate from the inherited round-332 raw-V name to
`orca1ice_tracer_stage3_kt2_np2`. Thus an operator can set
`R328_EXISTING_RUN` to a retained round-333 refusal and the inherited driver
will validate it, admit it, and move it back to the canonical round-333 target
without another NEMO run.

Every `--admit-existing` invocation, including the in-place canonical path,
now reads its expected producer from the run's single-entry
`producer_commit.txt` and its binary digest from `nemo.sha256`. Admission
requires the producer to be a local commit object and an ancestor of the
admission commit, and hashes the run's `nemo` against that manifest. The
in-place path no longer substitutes current `HEAD` for the producer or depends
on a currently staged target-build executable.

The read-only preflight constructs a disposable, manifest-pinned round-333
run and passes it through both paths: first in-place at the canonical name,
then through `R328_EXISTING_RUN` under `refused/` with the round-333 refusal
basename. Its terminal control is
`R333_SYNTHETIC_RETAINED_ADMISSION_PASS routes=in-place,retained`. The complete
preflight remains green: 77 inherited tests and 6 round-333 tests, all seven
placement plants, both-rank bounds/sign fixtures, 140-file deterministic
passivity, eight Fortran syntax proofs, 1,367 floating-point instructions on
both object arms, and 409/421 identical same-program units. It ends at
`ORCA2_ROUND333_TRACER_PREFLIGHT_READY`; no NEMO executable ran.

The exact ORCA2 landing battery passes 152 tests in 1,139.58 seconds. The
per-receipt citation gate still passes 8 citations with zero failures, zero
unmapped citations, and zero map-audit failures; its self-test plants fire.
No DINO month run is required because no model or package file changed.
Codex review: operator. The two reviewer HOLD items are dispositioned here;
operator rereview remains required before acquisition.

Disposition remains **ACQUISITION_NEEDED** at the same operator command and
expected evidence directory above. This resume makes both recovery paths
usable; it does not admit or synthesize the missing tracer record.
