# Split-explicit momentum chain: rounds 28--32 result

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Production result through row 1.4

Round 28 built the complete literal EEN coefficient path at commit
`27a3682209688bd872f448965ecdb5b65f10b1a1`. The bridge carries the raw NEMO
F-point Coriolis, U/V/F thicknesses, masks, reference depths, and all eight
horizontal metrics. `_nemo_literal_een_coefficients` at
`barotropic_latlon_cgrid.py:692-820` materializes the four U and four V corner
coefficients in the executed NEMO association: triads
`dynspg_ts.F90:1517-1528,1544-1555`, surface-to-bottom reduction
`:1530-1534,1557-1561`, and post factors `:1535-1538,1562-1565`.
`een_barotropic_coriolis` consumes that object at
`barotropic_latlon_cgrid.py:905-980`; both the pre-loop and live substep paths
share it. The selector is faithful by default only on `nemo_dino_kamm` and
`nemo_dino_kamm_mlf` (`dino.py:1470`); all other cards retain the generic byte
pin (`state.py:1132`). Missing operands and unknown selectors fail closed.

Artifact
`/tmp/dino_split_explicit_momentum_chain_round28_literal_builder.json`,
SHA-256
`0e8757a2c10cd99cbb8394f639e15beb84c05b244f4eeb349718785763e9106e`,
classifies `PRODUCTION_LITERAL_EEN_COEFFICIENTS_AT_BAR`. All eight coefficient
RMS errors are zero or `2.82e-18`; applied U is exact and applied V is
`1.29e-19` RMS with `1.28e-17` maximum/NEMO-RMS.

The unchanged production replay,
`/tmp/dino_split_explicit_momentum_chain_round28_production_replay.json`,
SHA-256
`1b4e83ec06416711cfc2b866c878cccb2a23988e8953c9b440a9e94c22bc2be9`,
puts row 1.3 SSH/U/V at `1.90e-20`/`2.08e-16`/`2.01e-16` and all five row-1.4
outputs between `3.26e-16` and `4.78e-16`: every registered field is AT BAR.
The target reduction from the old roughly `2e-7` row-1.3 production error is
therefore confirmed.

Focused bridge, config, JIT/gradient, literal-builder, AL81
energy/enstrophy, and conservation tests pass (157 passed, one expected
failure in the broad physics set; the later focused routing set adds 37
passes). The sole strict-suite failure is pre-existing and unrelated: the
stale `grid_type` allowlist in `test_validate_strict_coverage.py`.

## Ordered rows 2 and 3

Round 29 artifact
`/tmp/dino_split_explicit_momentum_chain_round29_rows2_3.json`, SHA-256
`3cd09d6bf3f7f5bb9ee51705a1e5e8cbf3671ff19cec6ebf407efd9e883d60d5`,
scores the second `div_hor` call at `stpmlf.F90:349-376` /
`divhor.F90:172-181` as its registered `CEILING`: production normalized RMS
`8.06495e-5`, zero shift uniquely best. The source-ordered offline divergence
is `3.44234e-5`; this is a transport/metric association ceiling, not a new
physical-term owner, and it legally releases row 3.

The second `dom_qco_r3c` call (`stpmlf.F90:378-394`,
`domqco.F90:153-185`) is the next formal stop. T/U/V/F normalized RMS values
are `3.3185e-16`, `3.0750e-16`, `2.9791e-16`, and `2.6900e-16`, but their
strict maximum/NEMO-RMS values are `2.07e-15`--`2.50e-15`; all four classify
NEAR-CLASS under the unchanged pointwise bar.

Round 30's preregistered 2^3 factorial binds production versus NEMO
`pssh_final`, vector versus source-ordered reference-depth reductions
(`domain.F90:139-160`), and live division versus materialized NEMO metric/post
factors (`domhgr.F90:144-160`, `domqco.F90:165-181`). Artifact
`/tmp/dino_split_explicit_momentum_chain_round30_qco_factorial.json`, SHA-256
`fe2e8023c176d455ba98253840d68eef76fb50ec2ad8c8d5fa03938fd1d61ccc`,
classifies `INHERITED_FROM_ROW_1_4`:

* depth association is inert;
* literal metric/post-factor association removes only an independent
  face-operation roundoff (U/V/F remain AT BAR under oracle SSH either way);
* substituting the unchanged NEMO `pssh_final` makes T exactly zero, and the
  fully literal P1D1M1 arm makes all four outputs exactly zero.

Thus QCO depths and local composition are exonerated. The strict row-3 residue
is deterministic amplification of row-1.4 `pssh_final`, which is itself AT
BAR under its registered accumulation gate (`3.26280e-16` normalized RMS,
`1.94207e-15` maximum/NEMO-RMS). Bars are not relaxed: row 3 remains the
ordered stop.

## Refuted update candidate and held next measurement

Round 31 tested the already-held round-26 identity for the explicit momentum
commit (`dynspg_ts.F90:700-705,719-732`) in production. The literal selector
changed none of the row-1.3 or row-1.4 metrics, failing the preregistered 90%
improvement bar. The candidate and tests were removed at
`4baa531ef641ce4a4cb965c53678c2626f828ec2`; no inert physics knob survives.
The binding replay is
`/tmp/dino_split_explicit_momentum_chain_round31_production_replay.json`,
SHA-256
`7ce94a68528e0e1b0fe950eeaa1ca194bcd9bbdfa0c8545459bbd7141888643d`.

The first missing discriminant is now the 68-substep SSH/U/V trajectory: does
the at-bar first-substep velocity residue enter SSH during the recurrence, or
only during final boxcar accumulation? Round 32 preregisters and supplies the
three-stream full-halo SLOT instrument, a 223/223+3 exact bracket, a
checkout-local CPU/fp64 scorer, and planted controls in
`dino_split_explicit_momentum_chain_round32_handoff.md`. It is DESIGNED, NOT
RUN. No NEMO, GPU, or MPI process was launched here.

Current ordered disposition: rows 1.1--1.4 are owned and at their registered
bars; row 2 is at its registered ceiling; row 3 is locally exonerated but
formally blocked by inherited row-1.4 last bits. Rows 4 (`dyn_zdf`), 5 (`wzv`),
6 (`mlf_baro_corr`), then the free-surface filter, momentum RHS, tracer tail,
and remaining registry chains are ORDERED-BLOCKED pending the round-32 held
trajectory.

Round-32 execution addendum: the host completed the build, both sequential
arms, and the exact 223/223+3 bracket. The first scorer attempt is invalid and
makes no science claim: it intercepted `scan` while the faithful card executes
`fori_loop`, and it treated the pre-update trace as post-update. The corrected
scorer tees the 68-row `fori_loop`, compares carry-in to NEMO substep-start,
and binds trace rows 1/2 to the existing entry/post-substep-1 dumps. Only the
scorer block in `dino_split_explicit_momentum_chain_round32_resumed_handoff.md`
must rerun; the NEMO artifacts remain admitted.

## Official round-32 closure and ordered promotion

The official corrected scorer artifact is
`/tmp/dino_split_explicit_momentum_chain_round32_substep_trace.json`, SHA-256
`0b284d7c8880646850daee86b2b85fb13fb65540402e2e9a02760b0ed60ba86f`.
It binds commit `588cbae24e8605008cf3ccac5dff55fb9c5460a9`, the admitted NEMO source,
binary, patch, bracket, and three trace streams, and classifies
`NO_STRICT_SSH_FAILURE`. There is no first strict SSH, U, or V failure in any
of the 68 split-explicit substeps. The final U/V point errors are
`4.83e-16`/`4.44e-16`, below the registered `1.4e-14` linear-accumulation
bound by a scalar factor of about 30: the remaining last-bit errors cancel
rather than accumulate. The actual substep-1, full 68-row capture, entry and
post-substep-1 alignment, `fori_loop` restoration, identity, and five-
`nextafter` planted controls all fire. The retained four-`nextafter` plant is
documented as structurally insufficient and is superseded by the firing
five-`nextafter` plant.

This receipt certifies the split-explicit recurrence end to end. Rows
1.1--1.4 and row 2 are promoted from the ordered gate. Row 3 is also promoted
as `INHERITED-BOUNDED`: its local QCO depth and composition operands remain
exonerated by the round-30 factorial, and the certified recurrence proves that
the inherited at-bar SSH last bits never create a strict downstream failure.
This is an ordered-gate release, not a relabeling of the row-3 NEAR-CLASS
pointwise statistic or a relaxation of its bar. The next executable registry
row is row 4 (`dyn_zdf`), followed by row 5 (`wzv`), row 6
(`mlf_baro_corr`), the free-surface filter, momentum RHS, and tracer tail.

## Rounds 33--34 retraction: wrong row-4 representation boundary

The round-33 artifact and round-34 operand table are invalid for science and
their executable scorers now refuse to print a verdict. NEMO's dumped stage-8
`naa_B` is already barotropic-free. Round 33 injected that field at the public
`_apply_implicit_vertical_mixing` entry, where production
`zdf_baroclinic_only` subtracts the mean again
(`ocean_model_latlon_cgrid.py:7275-7296`) and later re-splices the saved mean
(`:7550-7557`). It therefore double-stripped the fed field and compared a
post-splice result with NEMO's pre-splice `dyn_zdf` output. Round 34 captured
that same invalid arm, so its apparent RHS owner was a harness artifact.

The narrow controls remain factual but do not rescue either verdict: the S17
reconstruction selects `rDt=5400 s` and reconstructs NEMO's pre-stress
operands exactly, and the independent ZDF chain-end receipt proves the literal
dispatch kernel is exact for the oracle pack. The corrected round-35
measurement observes the unmodified production step's raw literal-dispatch
input and return, before mean readdition, and compares like representation
with like. Rows 4 onward remain open pending that preregistered result.

## Official round-35 row-4 boundary

The corrected artifact is
`/tmp/dino_split_explicit_momentum_chain_round35.json`, SHA-256
`4b6edffaafb7e3f079f8620fd899698c7b5f324ba00bedf1603ad4e2e1c94dbe`.
It runs one unmodified production day-180 step at commit
`cb9c3342848`, captures exactly the U then V raw dispatch calls, restores the
hook, and passes the same-JAX-path identity, finite-capture, wrong-`rDt`, wet
NaN, roll, RHS plant, and output plant controls. Its disposition is
`ROW4_LOCALIZED_TO_RHS`:

| ordered operand | U normalized RMS | V normalized RMS | disposition |
|---|---:|---:|---|
| selector / `rDt` / wet mask | 0 | 0 | exact |
| assembled RHS | `7.44406e-3` | `1.25751e-5` | first DEBT |
| face `avm` | `1.95309e-4` | `1.99454e-4` | ordered-blocked |
| cell thickness | `2.98257e-6` | `2.54150e-6` | ordered-blocked |
| interface thickness | `3.95588e-3` | `3.95535e-3` | ordered-blocked |
| drag diagonal | `6.58318e-6` | `4.78768e-6` | ordered-blocked |
| raw dispatch output | `4.25523e-2` | `1.33955e-2` | ordered-blocked |

The coefficient rows are not owners merely because they are red: the ordered
walk stops at the RHS. NEMO forms the baroclinic RHS and bottom correction at
`dynzdf.F90:137-178`, then deposits the surface stress at `:353-363` after
the barotropic removal. Production currently lets the explicit wind enter the
state before its baroclinic strip (`ocean_model_latlon_cgrid.py:7160-7180,
7275-7296`). Round 36 preregisters the additive placement decomposition before
changing that ordering.

## Official round-36 RHS peel and next stop

The registered additive artifact is
`/tmp/dino_split_explicit_momentum_chain_round36.json`, SHA-256
`f680200f2a3733558d1de7be7f97f5a80e575c003cbebf554fc1aa80e4428270`.
It binds commit `30c7cc2c746`, reconstructs its captured baseline from the
no-wind and stripped-wind terms at the accumulating bar, and passes every
identity, restoration, V-zero, roll, NaN, and planted-bar control. The result
is `WIND_PLACEMENT_REFUTED`:

| arm | U normalized RMS | V normalized RMS |
|---|---:|---:|
| production baseline | `7.44406e-3` | `1.25751e-5` |
| no-wind RHS | `6.74635e-3` | `1.25751e-5` |
| wind deposit | `3.80291e-6` | exact structural zero |
| delayed actual wind | `6.47452e-3` | bit-identical to baseline |
| delayed NEMO wind | `6.47446e-3` | bit-identical to baseline |

Delaying wind removes only `0.130243289` of the baseline U error, far below
the preregistered `0.90` majority bar. The already-carried centered-wind fix
is confirmed independently by the `3.8e-6` deposit residual; neither its
arithmetic nor its position owns row 4. The remaining first operand is the
no-wind pre-`dyn_zdf` momentum RHS.

The existing `acc_momentum_budget.py` was tried first, as required, but its
run is INVALID and prints no term verdict. Before reaching D03--D06 it applies
an old whole-ZDF uniformity control requiring the un-restored top/bottom RMS
ratio `<3.0`; the current faithful epoch measures `3.26` and the probe stops.
That bar is not relaxed post hoc. Round 37 preregisters a scoped U/V
D03--D06 accumulator scorer that retains the relevant time-level and closure
controls without importing the unrelated failed ZDF control. This extension
is the too-large instrumentation stop for this round. Rows 5 (`wzv`), 6
(`mlf_baro_corr`), free-surface filter, remaining momentum RHS, and tracer tail
remain ordered-blocked at row 4.

## Round-37 held scoped U/V term ladder

Round 37 is **DESIGNED, NOT RUN**. Enumeration of the admitted day-180
deterministic-writer inventory found full-halo D03--D06 cumulative U/V
streams and dedicated KEG, ZAD, VOR, LDF, and HPG increments already present.
No new NEMO writer, Fortran patch, build-tree copy, GPU run, or MPI run is
therefore justified. The committed producer instead captures production
public momentum diagnostics twice on CPU/fp64 and the bracket requires all
15 files to be byte-identical before scoring.

The exact, source-ordered ownership ladder is vertical advection/ZAD,
vorticity/Coriolis, lateral friction, the KE-gradient+HPG group, then the D06
total. Individual KEG and HPG partitions are diagnostic partials and cannot
own the interval alone. Every comparable 3-D term uses the unchanged
accumulating `1e-12` class bar. Full-halo NEMO streams are admitted by the
retained manifest; native twin staggering and the cited U/V interior crop are
checked explicitly. Roll, sign, wet-NaN, and `2x` plants must fire for both
components. The four-block producer/capture/bracket/score handoff is in
`dino_split_explicit_momentum_chain_round37_handoff.md`. Row 4 and all later
registry rows remain ordered-blocked until its held receipt is returned.

## Round-37 stopped-capture coverage adjudication

The first round-37 capture is INVALID for term ownership and emitted no
metadata. Its closure stop is a real Rule-1 coverage finding in the public
diagnostics: the explicit external surface-stress tendency had no diagnostic
slot. The remainder is confined exactly to the U surface level (9,793
nonzeros, maximum `1.9218384941372795e-5 m s-2`); deeper U levels and the
entire V field are bit zero. DINO's analytic stress is zonal-only, matching
round 36's independent exact-zero V wind receipt.

The active NEMO audit confirms this is not an omitted D03--D06 tendency.
`dyn_adv`, `dyn_vor`, `dyn_ldf`, and `dyn_hpg` are the active writers at
`stpmlf.F90:309-328`; optional damping, assimilation, boundary, AGRIF, and
OSMOSIS writers are inactive. Surface stress is applied only later at
`dynzdf.F90:353-363`. The twin diagnostic contract now carries the exact
surface-stress arrays used in the tendency update, with a two-component wind
closure test that is red without them. Round 37 records stress as a
coverage-only nonowner and leaves its source-ordered D03--D06 ladder and bars
unchanged.

The two headline magnitudes must not be divided: `1.9218e-5 m s-2` is a
pointwise absolute tendency maximum, whereas row 4's `7.444e-3` is a
dimensionless normalized RMS error. The former explains the stopped closure
gate; it does not reassign the row-4 residual, and round 36 already refuted
wind ownership at 13% removal. Rows 4 onward remain held pending the corrected
round-37 capture/bracket/score cascade.

## Official round-37 RHS ladder

The corrected scoped ladder completed with artifact
`/tmp/dino_split_explicit_momentum_chain_round37.json`, SHA-256
`053bafb547e66aa09a2f0a357c8bce3da5ba808689e3654b726e77d7f0ee0f23`,
and exact duplicate bracket
`bce8c5a037690487d7149010ab99826c7c498df202bc90da3b5a44a6c668b062`.
All closure, dedicated-increment, identity, roll, sign, wet-NaN, and two-bar
controls fire. The first failing exact D03--D06 term is vertical advection
(ZAD): U normalized RMS `1.4248796516210466e-5`, V
`5.7019720173785526e-5`. The later vorticity, lateral-friction, KE-gradient +
HPG, and D06 rows remain ordered behind it. This localizes the earlier row-4
assembled-RHS error; it does not equate the ZAD metric with the larger
`7.444e-3` raw-RHS boundary metric.

NEMO's active stock operator is `dynzad.F90:83-119`: zero surface carry,
surface-to-bottom `jk=1..jpk-2`, `e1e2t*ww(jk+1)` neighbour sums, Kmm velocity
shear, live `e3u/e3v(Kmm)` division, carried interface term, and a carried-only
bottom cell. `stpmlf.F90:275` computes the `ww` consumed by `dyn_adv` at
`:309`; the post-split call is not this operand.

## Rounds 38--39: ZAD is a QCO `ww` x face-thickness composition

Round 38 replayed the previously known `ww` inheritance at the exact day-180
round-37 state. Artifact
`/tmp/dino_split_explicit_momentum_chain_round38.json`, SHA-256
`51737434a48a2c625d0ac5472ab4a5914eca32a4a5d615034abb57ec993fb79b`,
reproduces the retained production diagnostic and a direct production-kernel
call bit exactly. Kbb/before velocity is decisively worse (`3.34e-3/3.44e-2`),
and NEMO call 2 is worse than the structurally correct call 1. Call-1 `ww`
reduces U error from `1.42488e-5` to `4.17258e-6` (70.716%) and V from
`5.70197e-5` to `6.08806e-6` (89.323%). It is causal, but neither component
meets the preregistered 90% majority rule, so the historical one-operand
`wzv` ownership claim is retracted for this matched state.

Round 39 then ran the registered 2^3 retained-input factorial over call-1
`ww` (W), source-literal live `e3u/e3v(Kmm)` (H), and explicit
`dynzad.F90:83-119` association (A). Artifact
`/tmp/dino_split_explicit_momentum_chain_round39.json`, SHA-256
`7104c30ad692241f13c828ecd75eadaee284762e64b619d45243471c35b18748`,
binds exact Kmm U/V and exact wet T/U/V metrics and passes every registered
control. Results are decisive:

| arm | U normalized RMS | V normalized RMS |
|---|---:|---:|
| production W0H0A0 | `1.42488e-5` | `5.70197e-5` |
| H only W0H1A0 | `1.51413e-5` | `6.07680e-5` |
| W only W1H0A0 | `4.17258e-6` | `6.08806e-6` |
| W + H W1H1A0 | `2.55945e-16` | `6.08581e-16` |
| W + H + A W1H1A1 | `2.33481e-16` | `5.98622e-16` |

H conditional on W removes `0.99999999994/0.99999999990` of the remaining
U/V error. A is inert at roughly `1e-17` effect. H alone makes the production
result worse, while W + H closes both components at the accumulating `1e-12`
bar. The owner is therefore the **interaction of NEMO QCO call-1 `ww` and live
Kmm face thickness**, not velocity timing, call timing, recurrence, source
association, or partial-cell bottom handling. The older divisor-only
refutation remains correct in its one-factor context; it concealed this
cancelling pair.

### Coupled fix design and ordered stop

A thickness-only patch is forbidden by the measured cancellation: it is
faithful locally but worsens both production rows. The required production
change is one coupled selector, provisionally `zad_qco_evaluation`, with no
independently selectable W/H half-arms:

1. at the DINO MLF step entry, predict Kaa SSH from the same current-volume
   continuity used by NEMO `ssh_nxt` (`stpmlf.F90:248`);
2. use Kbb/Kmm/Kaa SSH to evaluate the literal QCO `wzv` recurrence
   (`sshwzv.F90:218-227`);
3. build live Kmm U/V face thickness through the already-established raw-mesh
   `dom_qco_r3c` operands;
4. pass both operands together into ZAD before assembling the slow momentum
   RHS, preserving the current public diagnostic and autodiff contracts.

The selector must be faithful by default on only `nemo_dino_kamm` and
`nemo_dino_kamm_mlf`; all other cards must be byte-pinned. Validation requires
unknown/dependency-red config tests, synthetic source-literal W/H tests, JIT
and gradient coverage, diagnostic closure, the round-39 full-arm day-180
target, then round-37 and raw row-4 replays. This is an integrator/state-
contract restructure rather than a safe local fix for this round.

Ordered disposition: row 4's first failing term is fully owned by the W x H
composition but remains unfixed; row 5 (`wzv`) is the same architectural
operand and is open. Row 6 (`mlf_baro_corr`), free-surface filter, later
momentum-RHS rows, and tracer tail remain ordered-blocked. No held run is
needed: the next action is the coupled production implementation above.

## Round 40: production coupled QCO ZAD path

The production implementation is complete at commit `ef0cfed87ecf`; the
official replay artifact is
`/tmp/dino_split_explicit_momentum_chain_round40.json` (SHA-256
`723fe0e74724febab71537ef31cde16e32388c9da01c9058ef2c37629067258b`).
It classifies `ROW4_ZAD_AT_BAR`: U/V normalized RMS are
`2.55945e-16/6.08581e-16` under the unchanged `1e-12` accumulating bar. The
reconstructed call-1 `ww` and live Kmm U/V face thicknesses each have exactly
zero error against their retained NEMO operands. Identity, sign, roll,
two-bar, and thickness-only-worsens controls all fire.

`zad_qco_evaluation="nemo_literal"` is one inseparable path on the two DINO
fidelity cards and stays `generic` elsewhere. It left-accumulates Kmm
`divhor`, predicts Kaa SSH over the leapfrog span, applies the QCO WZV
recurrence, and supplies the same live Kmm face thicknesses to dynzad. The
final wiring defect was a 2-D land mask broadcast over depth at the production
call site; passing `OceanPartialCellCoordinate.is_active` restored NEMO's
dummy/below-bottom mask and closed the row. The bridge now also carries raw
`e2u/e1v`, so the literal continuity path does not reconstruct those operands.

Ordered disposition: row 4 is promoted. Row 5 is **not** promoted by this
receipt: round 40 certifies the call-1 WZV consumed by dynzad, while the
registered row-5 check is the post-dyn_zdf call-2 recurrence. Capturing the
corresponding production post-ZDF W state is the next held instrumentation
boundary. Row 6 and the free-surface-filter, momentum-RHS, and tracer-tail
chains remain ordered-blocked; no call-1 result is relabelled as call 2.

## Round 41 held: post-`dyn_zdf` WZV call 2

Round 41 preregisters the distinct row-5 boundary at NEMO
`stpmlf.F90:396,411-412,578` and the QCO bottom-up recurrence at
`sshwzv.F90:198-228`. The existing full-halo
`wzv_dump_ww_call2.bin` is reused; call 1 is retained only as the wrong-order
control. No new NEMO writer is needed. The canonical call-pair writer's unit
9103 is admitted by a whole-tree collision scan, including the mirrored
DINO/DINO_DBG MY_SRC/WORK copies.

The legoESM instrument executes the committed single-pass `_nemo_mlf_step`
verification path, captures each native `(199,52,37)` interface-W assembly,
and selects the unique array whose adjacent-interface average is byte-exact
with the `(199,52,36)` W state observed at the post-implicit-solve,
pre-`mlf_baro_corr` boundary. This closes the time-boundary receipt without
padding an interior array. Two fresh CPU captures, a byte-exact bracket, and
the registered accumulating `1e-12` score are held in the round-41 SLOT
handoff. Until that official receipt runs, row 5 stays open and row 6 plus
the later free-surface, momentum-RHS, and tracer-tail chains stay
ordered-blocked.
