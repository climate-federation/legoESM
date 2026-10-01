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
pin (`state.py:1147`). Missing operands and unknown selectors fail closed.

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
(`ocean_model_latlon_cgrid.py:7449-7470`) and later re-splices the saved mean
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
state before its baroclinic strip (`ocean_model_latlon_cgrid.py:7334-7354,
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

## Round 42: row-5 call1-to-call2 composition

The official existing-dump artifact is
`/tmp/dino_split_explicit_momentum_chain_round42.json`, SHA-256
`d9dec7def95b9dbe4389452f988ea2ea05087b4830b4af7dd1b90d3a8a441d29`.
Both endpoint reconstructions are bit-exact: H0Q0 reproduces call 1 and H1Q1
reproduces call 2 with normalized RMS `0`. The two single substitutions expose
a severe cancellation:

| arm | call-2 normalized RMS | call-delta squared-energy removal |
|---|---:|---:|
| H0Q0 (call 1) | `4.7717e-5` | `0` |
| H1Q0 (second `hdiv` only) | `3.33605e-4` | `-47.9514` |
| H0Q1 (barotropic Kaa `r3t` only) | `3.38725e-4` | `-49.4655` |
| H1Q1 (joint call 2) | `0` | `1` |

Thus neither half is a permissible fix: the `dyn_spg_ts`-corrected Kmm
transport divergence and barotropic Kaa thickness tendency are a coupled WZV
operand. The source audit also rejects the apparent `dyn_zdf` velocity owner:
`dyn_zdf` writes Naa, while call 2 reads Kmm. The corrected Kmm velocity was
installed earlier by `dynspg_ts.F90:1170-1174`; the second `div_hor` and Kaa
`r3t` were both materialized before `dyn_zdf`.

Disposition is `CALL2_HDIV_X_KAA_COMPOSITION`. Production must build the
literal second W using both operands together. Row 6 and later chains remain
ordered-blocked until the coupled production replay reaches the row-5 bar.

## Round 43: literal recurrence alone does not close production

Artifact `/tmp/dino_split_explicit_momentum_chain_round43.json`, SHA-256
`7109a7c0eae5684b778452cc5b07e4e1397a4b88849f37d448c515c451494eec`,
is valid and classifies `ROW5_WZV_CALL2_LITERAL_DIVERGED`. The coupled
call-2 selector, actual Kaa override, duplicate capture, post-`dyn_zdf` carry,
and red controls all passed, but normalized RMS remains `0.1712884891`.

Therefore the generic tracer-boundary velocity passed into the literal
recurrence is not NEMO's corrected Kmm velocity. Row 5 remains open. The next
ordered measurement separately scores actual Kaa `r3t` and the reconstructed
second `hdiv`; no further production algebra changes before that result.

## Round 44: production call-2 operand capture

Artifact `/tmp/dino_split_explicit_momentum_chain_round44.json`, SHA-256
`b2b38638bbd4e1e0aaf3f30e67ccec3998addb9af6057f9e2e12b53e76c2b515`,
passes the unique-capture, restoration, identity, roll, shape, and retained-
stream gates. Its ordered disposition is `CALL2_KAA_R3T_DIVERGED`: the Kaa
`r3t` passed by production has normalized RMS `4.37655e-6` and maximum error
`2.69191e-5` of NEMO RMS. The following Kmm `hdiv` is independently DEBT at
normalized RMS `0.226396` (correlation `0.977900`), but stays second in source
order.

RETRACTED after round 45: the first interpretation attributed the Kaa error to
the model's post-solver global eta-drift projection. Capturing the raw
pre-projection product leaves the metric unchanged, so that attribution is
false. Round 29's `3.2628e-16` `pssh_final` was measured in the registered
upstream-forcing-held frame; it cannot be cited as an unheld production result.

The H operand also has a named time-level mismatch. NEMO call 2 reads the Kmm
velocity rewritten at `dynspg_ts.F90:1170-1174`: entry Kmm plus
`un_adv*r1_hu(Kmm)-puu_b(Kmm)`. The current call passes the later tracer
transport velocity derived from `state_new`, which includes after-level
momentum composition. Round 45 tests the literal Kmm rewrite together with
the raw Kaa SSH; neither half is eligible to ship independently.

## Round 45: literal rewrite removes the structural row-5 error

The unheld production artifact
`/tmp/dino_split_explicit_momentum_chain_round45.json` is a valid strict stop,
not a promotion. The literal Kmm rewrite reduces H normalized RMS from
`0.226396` to `2.03569e-6`; the complete W error falls from `0.171288` to
`3.43293e-6`. Kaa Q remains `4.37655e-6`, exactly the round-44 value, proving
that the eta-drift hypothesis was wrong and that the residual enters through
the earlier split-explicit inputs.

This scale and frame match the already-owned row-1.1 forcing debt. The earlier
row-1.3/1.4 and round-32 certifications deliberately substituted NEMO's exact
assembled slow forcing before scoring downstream operations. Round 46 applies
the same committed hold and no other substitution. Row 5 remains open until
that registered conditional replay either reaches its unchanged bars or
exposes another local operand.

## Round 46: row 5 promoted in the upstream-exact frame

Artifact `/tmp/dino_split_explicit_momentum_chain_round46.json`, SHA-256
`a5c68a419877b61605e7cfa1e3e67caf0b5b8d0e77c32c2718e5e2fd71a252f2`,
classifies `ROW5_WZV_CALL2_AT_BAR_UPSTREAM_EXACT`. Only the registered
`zu_frc/zv_frc` boundary was held; seed, `Hu_avg/Hv_avg`, QCO geometry, Kmm
rewrite, hdiv, Kaa SSH, recurrence, and post-`dyn_zdf` carry were production.
Q/H/W normalized RMS are respectively `3.25755e-16`, `2.90617e-15`, and
`2.37700e-14`; their maximum errors over NEMO RMS are `2.50682e-15`,
`1.09795e-13`, and `5.26687e-13`, all within the unchanged `1e-12` bars.

The local row-5 owner is therefore the coupled raw Kaa plus source-literal
Kmm rewrite at `dynspg_ts.F90:1170-1174`. The unheld W residual
`3.43293e-6` remains an upstream row-1.1 qualification and is not relabelled
as local exactness. Row 5 is promoted conditionally and releases row 6.

## Round 47: row 6 stops on the pointwise maximum axis

Artifact `/tmp/dino_split_explicit_momentum_chain_round47.json`, SHA-256
`60a4b7d8f45b8cc9a611deb4d4e641e5952c67960f6edff2ddade51771a3e4aa`,
passes the retained-stream, forcing-hold, hook, null, identity, roll, planted-
point, nonzero-correction, shape, and restoration gates. The local arm feeds
NEMO's pre-correction Kaa U/V into production `_apply_after_level_reconcile`
with the model's upstream-exact primary barotropic target.

U/V normalized RMS are only `1.13036e-16/1.53239e-16`, but their maximum
errors over NEMO RMS are `2.98986e-15/5.63618e-15`, above the registered
POINTWISE `1e-15` bar. The ordered disposition is therefore
`ROW6_MLF_BARO_CORR_U_DIVERGED`; row 6 is not promoted. Round 48 separates
the few-ULP residual into primary-target and source-association operands.

Round 48 artifact `/tmp/dino_split_explicit_momentum_chain_round48.json`,
SHA-256 `8055e1138c7784942167ae06212e2801d78f6690d3646351b6040d557a70c06b`,
classifies `OPEN_UNRESOLVED`. Substituting NEMO's primary target removes
`82.206%` of joint squared error, while replacing the shared tree reduction
with the algebraically cancelled reference-ladder left reduction removes
`-0.504%`. The joint arm still has maximum/RMS `2.98986e-15/2.81809e-15`
for U/V. Thus the target is material but not sufficient, and the simplified
reference cancellation is not the remaining arithmetic owner.

The source shows why: `stpmlf.F90:744-755` actually materializes live
`e3u/e3v(Kaa)`, left-accumulates transport, then multiplies by live
`r1_hu/r1_hv(Kaa)`. The QCO factors cancel algebraically but not before
rounding. Round 49 tests that executed association from existing Kaa r3 dumps.

## Round 49: row-6 local arithmetic owned; production change held

Artifact `/tmp/dino_split_explicit_momentum_chain_round49.json`, SHA-256
`501f15eb8b2893b2f04fd73e8256ab9d2099cec865b7673198e3f359718b7018`,
classifies `ROW6_LOCALIZED_TO_LIVE_QCO_ASSOCIATION_GIVEN_ORACLE_TARGET`.
With the oracle primary target fixed in both arms, the algebraically cancelled
C0 reconstruction retains U/V maximum-error-over-NEMO-RMS
`2.98986e-15/2.81809e-15`. Executing the live QCO composition in C1 makes
both components bit-exact: normalized RMS and pointwise maximum are `0` for U
and V. Identity, roll, wet-point, nonzero-live-factor, and round-48 C0
reproduction controls all fire.

The owner is the executed association at `stpmlf.F90:752-765`: initialize the
transport with live `e3u/e3v(Kaa)`, source-left accumulate levels 2 through
`jpkm1`, then evaluate the correction with independently materialized live
`r1_hu/r1_hv(Kaa)`. Although the thickness and reciprocal cancel
mathematically, replacing that execution with a cancelled depth mean changes
the final few ULPs. Round 48 remains an upstream qualification: substituting
the oracle primary target accounts for `82.206%` of joint squared error, so
the live-association result does not claim that production's target is exact.

Row 6 is therefore **owned but not fixed**. A faithful production change is
larger than this round because the raw Kaa QCO state is currently lost across
the model-only post-solver eta projection. The registered implementation must:

1. carry raw Kaa `r3u/r3v` (or equivalently the raw pre-projection Kaa SSH)
   from the split-explicit solver to `_apply_after_level_reconcile`;
2. materialize live Kaa face thicknesses and reciprocals and execute the NEMO
   left-reduction and post-factor order literally;
3. keep the primary-target residual separately visible as upstream debt; and
4. default faithful only on the two DINO cards, preserve generic/off cards
   byte-for-byte, and include red controls for cancelled association and stale
   Kaa state.

No held run is needed to establish this ownership. The next work item is the
cross-interface implementation above; row 6 cannot be promoted before its
production replay reaches the unchanged pointwise bar. The free-surface
filter, momentum-RHS tail, and tracer-tail chains remain ordered-blocked behind
that production certification.

## Round 50: row 6 certified on the production carry

Artifact `/tmp/dino_split_explicit_momentum_chain_round50.json`, SHA-256
`9d17606b1dfe4190456df58cb1a0e4aaac6e6456c86849fb93dbfc39f3b7dc80`,
classifies `ROW6_MLF_BARO_CORR_AT_BAR_UPSTREAM_TARGET_EXACT`. The actual
`_nemo_mlf_step` carries raw pre-projection Kaa SSH, rebuilds live native-face
QCO thicknesses and reciprocals, and executes `stpmlf.F90:752-765` in source
order. With the retained oracle primary target, U is bit-exact and V has
normalized RMS `1.88713e-20` and pointwise maximum/RMS `1.10082e-17`, both
inside the unchanged `1e-15` bar. The stale-Kaa and point/roll plants fire.

The production-target arm remains explicitly qualified: U/V normalized RMS
are `1.02615e-16/1.42064e-16`, but pointwise maximum/RMS are
`2.98986e-15/5.63618e-15`. This is the registered upstream primary-target
debt, not a local row-6 failure. Row 6 is promoted conditionally in the same
upstream-exact frame as rows 4--5 and releases the free-surface-filter chain.

## Round 51: free-surface filter chain is bit-exact

Artifact `/tmp/dino_split_explicit_momentum_chain_round51.json`, SHA-256
`aaa492d7107ba00096d1d93c9240127856365dae71f56fbe65f41f611e1146c1`,
classifies `FREE_SURFACE_FILTER_AT_BAR`. `ssh_atf` and the filtered
`r3t_f/r3u_f/r3v_f` are all bit-exact under the unchanged pointwise bar. The
correct Kaa operand is final `spg_dump_pssh_final`, not the pre-split
`ssh_nxt` state: substituting the latter gives SSH normalized RMS
`7.60141e-6` and maximum/RMS `8.33751e-5`, so the wrong-time-level control
fires strongly. The momentum tail is released.

## Round 52: momentum tail stops on the cancelling Kmm cycle

Artifact `/tmp/dino_split_explicit_momentum_chain_round52.json` classifies
`MOMENTUM_TAIL_DIVERGED_KMM_REWRITE_U`. `finalize_lbc` is bit-exact for U/V.
The first debt is NEMO's centered Kmm execute-and-undo cycle:
`dynspg_ts.F90:1172-1173` installs the transport mean for tracer advection and
`stpmlf.F90:787-790` removes it before `dyn_atf_qco`. Skipping the
algebraically cancelling pair leaves U/V maximum-error-over-NEMO-RMS
`2.98961e-15/2.81807e-15`; `dyn_atf_qco` inherits the same pointwise maxima.
The oracle-Kmm filter arm is at bar, so the filter formula is exonerated and
the cycle arithmetic is the sole local owner.

Round 53 production replay retains the stop. With production `Hu_avg/Hv_avg`,
the literal cycle gives U/V pointwise maximum/RMS
`2.98961e-15/5.63615e-15`; the filter inherits the same values. The next
source-ordered operand is therefore the transport average itself, for which
unchanged retained `spg_dump_{un,vn}_adv_final.bin` streams already exist.

## Round 54: momentum tail conditionally closed

Artifact `/tmp/dino_split_explicit_momentum_chain_round54.json`, SHA-256
`a1b76177330b83d7bb21c7f35c9e606d10d36a4b98d46b3b570f53588188eaa3`,
classifies `MOMENTUM_TAIL_AT_BAR_UPSTREAM_TRANSPORT_EXACT`. Substituting only
NEMO's retained `un_adv/vn_adv` into the production Kmm execute/undo cycle
makes restored Kmm U/V and consequent Asselin-filtered U/V bit-exact. The
round-53 production-transport rows remain DEBT and are retained as an upstream
qualification; this result closes the cycle and filter arithmetic, not the
production `Hu_avg/Hv_avg` operand. The tracer tail is released for a fresh
production-entry replay against its previously admitted held streams.

## Round 55: tracer entry exposes a missing Kmm write

Artifact `/tmp/dino_split_explicit_momentum_chain_round55.json`, SHA-256
`46aaf5f1644304151a9377ace037f4a4e03a91f1f0255885296af433e67e0f9d`,
stops at subrow 8.3: all `9758/9758` wet U columns diverge, correlation is
`0.9846254752`, RMS ratio is `1.0294306361`, and maximum normalized column
error is `2.6794216264`. This reproduces the old lateral-lane signature at the
current production epoch and is many orders above round 53's transport ULPs.

The missing operation is structural. NEMO installs the transport-corrected
Kmm velocity at `dynspg_ts.F90:1170-1174` before `traadv.F90:301-304` points
`zptu/zptv` at Kmm. legoESM used the same literal cycle for WZV call 2 and the
later Asselin filter but left tracer horizontal mass flux on its generic
`state_new` correction. Round 56 wires the execute half into tracer entry
under the already-scoped literal QCO selector; GM/FCT remain ordered-blocked
until that production replay passes.

Round 56 executes the production fix and reduces the maximum normalized
column error from `2.6794216264` to `1.9408056746e-5` (about `1.38e5`-fold),
with correlation `0.9999999999997489` and RMS ratio `0.9999999850`; the strict
pointwise bar still rejects `9758/9758`. Round 53's smaller tail was measured
under a held NEMO slow forcing and held after-level target, so it was not an
unheld transport replay. Round 57 therefore substitutes only the already-
registered `zu_frc/zv_frc` streams before reopening any downstream tracer
operand.

**Population correction before adjudication.** The first round-55--57 scorer
revision used nonzero `e3u` as its wet mask. That admitted `10,348` columns,
including 590 dry U faces; the registered NEMO `umask` population is
`9,758` columns / `336,338` elements. Those receipts remain useful as red
plants for the structural Kmm fix, but their column counts and maxima are
retracted for ownership. Round 58 reruns the unheld and held arms on the exact
captured 3-D U mask before any transport-accumulator change.

## Round 58: corrected population localizes the final tracer-entry ULPs

The corrected unheld artifact
`/tmp/dino_split_explicit_momentum_chain_round58_unheld.json`, SHA-256
`c114565363360439e155deec96882828e553887ba49b1bfbfd940c50c6998e29`,
retains the production stop on the exact `9,758`-column / `336,338`-element
umask: all 9,758 columns diverge and the maximum normalized column error is
`1.8702849959e-5` (correlation `0.9999999999997494`, RMS ratio
`0.9999999850425257`).

The registered slow-forcing arm
`/tmp/dino_split_explicit_momentum_chain_round58_held.json`, SHA-256
`c542be24a4a646ada9a2d2bddfa380dce1c05f75ca05d3c17214bed75c897f9c`,
removes nearly all of that error but does not clear the bar: `170/9,758`
columns remain red, maximum normalized column error `5.9792156327e-15`,
correlation and RMS ratio both `1.0`. All population, identity, wet-point,
roll, sign, substitution-count, finiteness, and hook-restoration controls pass.

Together with round 54's bit-exact oracle-`un_adv` substitution, this owns the
remaining local residual to the barotropic transport accumulator. NEMO forms
`un_adv += za2 * zhU * r1_e2u` / V analog at
`dynspg_ts.F90:734-737`, preserving the metric multiply and reciprocal, and
divides the completed sum once by `r1_wgt2s` at `:999-1000`. legoESM instead
accumulates normalized secondary weights against the algebraically cancelled
`H*U` transport (`barotropic_latlon_cgrid.py:1259-1260,2041-2047`). This is a
few-ULP association difference after the slow-forcing hold.

Row 8 remains OPEN at subrow 8.3; 8.4--8.10, Redi T/S, complete tracer ZDF,
and tracer Asselin remain ordered-blocked. The literal accumulator is the
genuine too-large boundary for this round because the same carry contract must
remain correct in differentiable scan, fori-loop, and wide-halo chunked paths.
No new held run is required; all operands already exist.

## Rounds 59--64: literal transport carry and live Kmm face thickness

Round 59's held production replay (SHA-256
`85ea27cce4804d98f281940fe472e798d9fa64c741c23bb55e3fca40ee9ca677`)
reduced row 8.3 to a 104-column, `1.4948039082e-15` association residual.
Round 60 (SHA-256
`b3ef5c0534ff1348dbdb581686aa602cc1d9eca9ef61336ca0b4130217e54e2d`)
proved that NEMO's raw `za2*zhU*r1_e2u` / V analogue, followed by one final
division, is exact on the retained operands.  The production implementation
therefore preserves that accumulation topology consistently in the scan,
fori-loop, and wide-halo chunk paths; generic paths retain the normalized
historical form.  The direct cycle then made 8.3--8.4 bit-exact and exposed
8.5, the live Kmm face thickness, at `7.4551634675e-5` in all 9,758 columns
(round 63 SHA-256
`3a1a25ca328761b1bcbeb87953751a3a15b1ac00852b2ff62fd4223d107d24e0`).

Round 64 (SHA-256
`8858d60a51b07181e290fead087e4bab69c0d15e271bbdecb7d458ba12d4e4c7`)
certifies rows 8.3--8.7 after threading the live Kmm QCO face thickness: rows
8.3--8.6 are bit-exact and 8.7 has maximum normalized error
`7.1212637199e-15`, inside its accumulating bar.  The first failure moves to
the GM increment, row 8.8 (`8938/9758`, maximum `3.8631299623e-7`).

## Rounds 65--75: GM coefficient ladder closes tracer entry

The ordered GM peel first localized row 8.8 to `aeiu`, then to the Treguier
`zn` precursor.  The initial carried-N2 result exposed two coupled geometry
requirements: raw `rn2b` and the complete live Kmm W thickness including its
surface value.  A partial surface carry worsened the registered arm and is
retained as a red control, not an ownership result.  Round 71 (SHA-256
`6500acfa930c0342430fd1e57cfb1da023b0978e8fda3561e6133ffe12368821`)
makes `e3w(Kmm)` and `rn2b` exact and reduces row 8.8 to
`2.4207639254e-11`; its first residual coefficient is `zaeiw`.

The post-chain and Rossby-radius factorials exonerate the `zRo^2*sqrt(zah/zhw)`
association, Coriolis reconstruction, and bounds, localizing successively to
`zRo`, `zn`, and finally the forward value of
`sqrt(MAX(rn2b,0))`.  Round 74 (SHA-256
`56db4716cba582654fbd7bb55178a699b55678a1afdea9d8d8fe3cc670eea6fb`)
shows that exact-zero forward sqrt closes `zn`, `zRo`, `zaeiw`, `aeiu`, and
row 8.8, while the historical `1e-30` floor remains red in 982 columns.  The
production selector preserves NEMO's exact forward zero with a custom finite
zero derivative for nonpositive inputs; only the two NEMO DINO cards opt in.

Round 75 (SHA-256
`750c40875300ddda48287d84089c8931eaaecc71e8aab6ce7f4e18a0c806edf4`)
classifies `TRACER_ENTRY_ROW8_AT_BAR_EXACT_GM_SQRT`.  Rows 8.3--8.6 are
bit-exact; maxima for 8.7--8.10 are respectively
`7.1213e-15`, `1.3503e-13`, `7.1287e-15`, and `3.6437e-15`, all inside the
unchanged accumulating bar.  The full tracer-entry chain is therefore closed.

## Rounds 76--78: Redi entry and live-W-thickness majority owner

Round 76 (SHA-256
`45e4f8afda737b41e457668fe1ab7cc28ded09d3f7be06fabdd15e9804936a76`)
is the first ordered tracer-tail measurement.  It stops at Redi temperature:
all 9,920 wet columns are red, maximum normalized error
`0.1165888037`, correlation `0.9999949375`, and RMS ratio `1.0000806132`.
Salinity is co-located red with maximum `0.1119039414`.  Tracer ZDF and
Asselin remain ordered-blocked.

Round 77 (artifact
`/tmp/dino_split_explicit_momentum_chain_round77.json`, SHA-256
`dcc0cff4c63b30024794ad25b023b65223fe87137e5052b6d7dc478066973d14`)
substitutes only NEMO's live `e3w(:,:,Kmm)` into both MSC uses:
`traldf_iso.F90:314-332`'s `akz` construction and
`traldf_iso_scheme.h90:126-129`'s explicit-A33 reciprocal.  Temperature's
maximum falls to `7.2152808253e-4` (99.381% removed; correlation
`0.9999999994435`) and salinity's to `5.9841977660e-4` (99.465% removed;
correlation `0.9999999995771`).  Both remain red in every column, so the
registered disposition is `REDI_MSC_E3W_MAJORITY`, not full ownership, and
the production selector is deliberately not promoted yet.

No retained stream contains the remaining Redi flux operands.  Round 78 is
therefore the current held frontier: six deterministic full-halo streams for
T/S `zfu`, `zfv`, and total `zfw_kp1`, scored in that NEMO order at the
pointwise bar.  Units `9450--9455`, zeroed halos, exact shared-stream bracket,
and the complete build/run/score cascade are frozen in
`dino_split_explicit_momentum_chain_round78_handoff.md`.  A `zfw` failure
releases the A31/A32-versus-A33 split; six passing fluxes with a red total
release the divergence/volume-factor association peel.  This is the first
measurement requiring a held NEMO run, so the ordered walk stops here.

## Rounds 79--93: Redi T/S closure and registry completion

Rounds 79--92 successively closed the Redi horizontal fluxes, vertical skew,
and post-stage W-slope ownership. Round 93's existing-dump factorial owns the
remaining A33 debt to NEMO's written `zahu_w*wslpi*wslpi` / V association and
lands the shared explicit/implicit literal builder. Temperature T.1--T.3 and
salinity S.1--S.2 are strict AT-BAR. Salinity S.3's sole remaining
`6.690652e-15` association residue is cleared as
`PROVEN-ORACLE-ARITHMETIC` under Rule 1b after every registered association
arm is exhausted; it is not relabeled AT-BAR.

The released tracer ZDF application reuses the bit-exact registered rows
30--32, and the final tracer Asselin endpoint retains its prior
verified-formula/structural-target qualification. The chain is complete. Full
receipts and the Rule-1b conditions are in
`dino_split_explicit_momentum_chain_round93_result.md`.
