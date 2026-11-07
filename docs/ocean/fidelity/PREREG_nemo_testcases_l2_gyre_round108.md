# Preregistration: NEMO-testcases L2 GYRE round 108 compiled `bn2` walk

Date: 2026-09-18. Frozen at incoming lane tip
`a91c48e11c2fae2d6d482dfceba3e7097cbede69`, before any round-108
measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round108/`.

## Ordered statement and compiled source

Round 107 found that the first non-bit operand of the compiled TKE matrix RHS
is inherited from `rn2`, not owned by the RHS. Through the production JIT,
`rn2` was unequal in 16,232 cells with maximum absolute error
`4.438452643612534e-19`; `p_avt * rn2` was unequal in 16,137 cells with
maximum `4.367513634279986e-22`. Production eager had only five unequal
`rn2` words, all `+0.0` versus `-0.0`, and no numerical difference. This
round therefore leaves the downstream RHS and walks the compiled `bn2`
producer in statement order.

The record-producing compiled program calls `eos_rab` and then `bn2` from
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/stprk3.f90:159-160`, copies the
result at `:162-163`, and later calls vertical physics at `:168`. The executed
`bn2` loop is `eosbn2.f90:1609-1619`: it writes the depth weight `zrw` at
`:1610-1611`, interpolates thermal and haline expansion coefficients into
`zaw` and `zbw` at `:1613-1614`, and writes masked `pn2` from the temperature
contribution, salinity contribution, gravity product, and thickness division
at `:1616-1618`. The expansion coefficients consumed by this statement are
written by the compiled S-EOS arm at `eosbn2.f90:1312-1326`.

The existing consolidated round-46/51 stage gate and its private production
trace are extended. No second harness, public option, or NEMO acquisition is
created. Each arm returns exactly one selected value through the production
step so that returning a tuple cannot itself change the tested lowering.

## Immutable trajectory arm and stage order

The immutable trajectory arm remains
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`. Its headline
rows are kt2 T/S/U/V `1.4210854715202004e-14` /
`2.1316282072803006e-14` / `2.7377110452773967e-12` /
`3.2849219221489645e-12`, kt3 T/S `1.627497246303733e-4` /
`6.327735185607253e-6`, and day-30 T RMS
`1.2397011296352737e-2` K.

Decision 41 still makes kt1 stage-1 U the global first owned non-bit output:
7,620 unequal cells at `5.421010862427522e-20`, still AT-BAR. This downstream
`bn2` walk is attribution work only. A correction cannot land while that
earlier stage output remains owned non-bit. If the stage condition has
unexpectedly changed, a candidate must instead pass the full 954-row Rule-12
ladder and days 1-30 before promotion.

## Frozen predictions and falsifiers

**P1 — reproduced input boundary and replay.** Before interpreting an
intermediate, the production walk will reproduce round 107's final `rn2` row:
16,232 unequal cells and maximum `4.438452643612534e-19` under production JIT,
and five unequal signed-zero words with maximum numerical error zero under
production eager. A source-ordered NEMO-from-NEMO replay built from the
admitted recorded step-entry fields and raw NEMO geometry will reproduce the
round-59 recorded `rn2` output bit-for-bit. REFUTED if any count, maximum, or
the replay differs.

**P2 — one-output compiled-order walk.** Nine separate arms will return one
selected value while still executing the production step: `zrw`, `zaw`,
`zbw`, the temperature contribution, the salinity contribution, their
difference, the gravity product, the thickness division, and masked `rn2`.
Every row will be scored in production JIT, production eager, isolated-closure
JIT, and isolated-closure eager modes against the same source-ordered
reference.

`zrw` is predicted BIT in all four modes because its three depth products are
already separated by the transcription's source-order barriers. `zaw` is
predicted to be the first non-bit row under production JIT and isolated-
closure JIT because the compiled interpolation writes two rounded products
before their addition, whereas XLA may contract or reassociate this first
unprotected multiply-add. Both eager modes are predicted BIT through the
numerical output; production eager may retain only the already known five
signed-zero differences in masked `rn2`. REFUTED if any row before `zaw` is
non-bit, if `zaw` is BIT in either JIT mode, if an eager row has a nonzero
numerical error, or if the NEMO replay does not close.

**P3 — first-statement ownership.** The first non-bit arm, not the final
`rn2` row, will name the owned statement. Its unequal-cell set must be carried
forward through later rows and the final error must be numerically explainable
from it. A later row cannot be named while an earlier row is non-bit. REFUTED
if the cell-set and magnitude propagation do not support the named boundary;
in that case the result is an unresolved operand/lifetime boundary, not a
candidate.

**P4 — optimized-code discriminator.** Only after P2 names a boundary, the
corresponding isolated arm may be compiled with optimized HLO/LLVM dumping.
The eager source association is predicted to show separately rounded products
and an addition, while the optimized representation is predicted to contain
a fused or reassociated path without that binary64 boundary. REFUTED if the
matched emitted function preserves the compiled source tree. A dump without a
module/function match is UNMEASURED and cannot justify a patch.

**P5 — plant.** Under production JIT, the scorer will advance one finite,
nonzero, previously equal reference cell in the selected first-exact row by
one ULP. Exactly that row must gain one unequal cell, the log must print
`STATUS PLANT-FIRED`, and the process must exit nonzero. REFUTED if the plant
uses zero, changes no cell or more than one cell, prints PASS, or exits zero.

**P6 — landing and magnitude.** No arithmetic candidate is preregistered.
This round first locates the statement; any replacement requires a frozen
addendum before measurement and bit-exact production-JIT stage proof. Even an
exact downstream correction remains HELD while kt1 stage-1 U is the earlier
owned non-bit. The `rn2` error is below the observed kt3 T and day-30
temperature gaps by many orders and is not claimed to own either. No private
instrument arm is trajectory-scored. REFUTED if the upstream stage order is
found exact; the full ladder and days 1-30 then become mandatory, with no
AT-BAR row leaving the bar, no earlier first-over-bar, and every moved row
registered.

**P7 — other configurations.** DINO shares the `bn2` statement and would need
a production-step before/after comparison for any unconditional correction.
LOCK_EXCHANGE and OVERFLOW are predicted not to instantiate this prognostic-
TKE path. ORCA2 remains UNMEASURED-WITH-SPEC and requires a one-step
production comparison before promotion. REFUTED for either small tank if its
resolved card executes the statement. No unmeasured card is called a pass.

**P8 — instrumentation neutrality.** With the new private selector empty, the
restored-production stage twin and focused tests will reproduce the incoming
tip. REFUTED by any default-path bit movement. Diagnostic code that is not
needed by the committed gate will be removed; no measurement arm becomes a
physics default.

## Measurement order and controls

1. Commit this preregistration.
2. Extend the consolidated gate, existing entry-`rn2` bundle, and private
   statement trace with one `bn2` intermediate selector and a fail-closed ULP
   plant; add direct tests.
3. Reproduce P1 from the admitted round-46/59 records before scoring P2.
4. Run the nine arms separately in compiled order and all four execution
   modes. Stop at and name the first non-bit statement; preserve every row.
5. Inspect matched optimized code only for the first boundary. Freeze an
   addendum before any candidate, or leave the result HELD without one.
6. Restore the default selector, run the kt1/kt2 recorded-entry and chained
   stage tables, then run the separate read-only Codex refutation pass,
   citation gate and shifted-citation plant, and focused tests. Quote terminal
   test summaries rather than trusting shell status.

## Scope limits

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified and neither
`makenemo` nor `mpirun` is run. No public configuration, default, scheme
selection, threshold, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, #1484 guard, or immutable trajectory
baseline is changed. No stabilizer is introduced. The held round-105 routing
and every other held manifest remain held.

## Frozen source-branch correction after the fail-closed selector check

The original ordered-source section incorrectly called
`eosbn2.f90:1312-1326` the executed expansion-coefficient arm. Before any
round-108 numerical row was emitted, the committed gate checked the
instantiated card, found `n2_eos_form="teos10"`, printed
`round-108 bn2 replay requires the compiled GYRE S-EOS arm`, and exited
nonzero. The failed log is retained as
`round108/bn2_masked_rn2_production_jit.log`; it is not measurement evidence.

The compiled and configured branch is `CASE( np_teos10, np_eos80 )` at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1257-1310`, with
GYRE's resolved `ln_teos10=.true.` and `ln_seos=.false.` in
`EXP00/namelist_cfg:126-128`. The corrected replay uses that 35-coefficient
ALP/BET Horner program, including `SQRT(ABS(S+rdeltaS)*r1_S0)` and beta's
`/zs`, then enters the unchanged `bn2` loop at `:1609-1618`. P1-P8 and their
falsifiers remain frozen; only the erroneously named upstream input producer
is retracted and replaced before rerunning the refused gate.

The first corrected-branch replay was also refused before production scoring:
it was numerically exact but differed from the recorded final output in five
dry `+0.0`/`-0.0` words because it reconstructed `pab` and tracer differences
from legoESM's below-seafloor-extrapolated T/S rather than the raw NEMO entry
T/S promised above. That refused run is retained as
`bn2_masked_rn2_production_jit_v2.log`. The gate now passes the raw recorded
T/S and replay-certified TEOS-10 `pab` as the given-input arm and reports the
ordinary model-upstream arm separately. This is an input-bridge correction,
not a new arithmetic prediction; P1 still applies to the ordinary chained
arm and P2 to the given-NEMO-input arm.
