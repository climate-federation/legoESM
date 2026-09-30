# NEMO testcase L2 GYRE phase 3 — round 120 year gate and KEG materialization receipt

Date: 2026-09-19

Status: **HELD — Decision-45 gate and diagnostic landed; no physics or
configuration changed.**

## Verdict

Decision 45 is now fail-closed in the existing Decision-43 gate.  A candidate
year member must be an admitted clean fp64 seed-0 360-day run with the exact
from-rest cadence, all eight registered days are scored by the existing
day-gap implementation, the month and year day-30 rows must agree, and days
240 and 360 must not worsen.  The immutable year arm reproduces all eight
published rows exactly.  A one-ULP worsening at day 240 makes the command
print `STATUS PLANT-FIRED` and exit nonzero.

The Round-119 production-JIT KEG split is also closed as an attribution
question, but it is not a physics candidate.  At the sole active U target,
the unmasked, face-masked and WRITE-only KEG addends are bit-identical.  Even
so, only the private **masked-materialized** full-production-step arm makes
the after-KEG, after-ZAD and after-ADV boundaries bit-identical to the isolated
reconstruction.  Unmasked materialization and literal subtraction are
bit-identical to the baseline arm and retain the one-cell, one-ULP split.
Production eager has no split under any arm.  The frozen prediction that
unmasked materialization would close is therefore **REFUTED**; the measured
classification is **mask materialization under the complete production JIT**,
not different mask arithmetic at the target and not an error in NEMO's KEG
statement.

The effect is one `4.1359030627651384e-25 m s-2` word, while the admitted
day-240 T3D RMS gap is `1.6446741930292448e-2 K`.  No material source-exact
production candidate was preregistered, so the ladder, month, DINO and
candidate-year trajectories were correctly not spent.  No known gap is
declared closed: W remains the first and largest measured ZAD operand, but how
much it carries at day 240 is **UNMEASURED**.

## Frozen provenance

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round120.md`, SHA-256
`57b27de95eb5254687fb7b37d836dfcae0474251310ca890d4c688a6c17984fe`,
committed before measurement as
`c714a79ed7ccbe7fb4e99e9dd58985a77568364e`.  The clean scientific
measurement commit is `8469aaf8dc966d3379cb04364a3ed41dd25d0461`; the
corrected identical-graph plant commit is
`6ea869998c01f0e65bd7307e806f690447cd8663`.  The gate implementation and
its CLI plant test were committed first as
`441f72de4652cd21a2e12a17df8886af540e73df` and
`d1dfc44871669afc28fd0416b5591effd367d75e`.

The diagnostic extends
`nemo_testcase_l2_gyre_round83_slow_forcing_walk.py`; it does not create a
second stage harness.  The year admission extends
`nemo_testcase_l2_gyre_decision43_gate.py` and delegates numerical scoring to
the committed year-owners day-gap implementation; it does not create a second
year scorer or change the from-rest harness.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round120/`:

| artifact | role | SHA-256 |
|---|---|---|
| `year_gate/before_year_rows.json` | admitted immutable baseline year rows | `3d4107741937473aa8e3dbb6d68fc9eb9fd3a7b33657090ec6e8fb89580d50a5` |
| `year_gate/before_year_rows.log` | exact scorer table and PASS verdict | `4a269ef9bbd9db59b95e6b963d06c2a60614fe95a3084a2ee49ba68346d34e68` |
| `association/production_jit.json` | authoritative complete-production-step discriminator | `8055336be2471483cad2adfd0cac5bde9559782adbfac1dd62ab9436ee01f68a` |
| `association/production_eager.json` | separately labelled eager control | `f4ebd0080710d59c6e32d05bd58462b3432b32574d606f122f2b3ef39d0e7068` |
| `association/keg_ulp_plant.json` | corrected identical-graph production control | `27a125b8d3140de24271ddb7c31e386ddc3aebca604bd267de70727e77ccccff` |
| `association/production_jit.log` | JIT gate log with frozen-prediction refusal | `e945ad711db02c2628cc201e0bf7d96937ae5ed8f4cb03edaac7897b9af6636c` |
| `association/production_eager.log` | eager gate log | `5c88fba1a8221003b69906222f6c205a5fd80314b8eef76decc53477f087f2bb` |
| `association/keg_ulp_plant.log` | named nonzero plant verdict | `9c6c4723e414488c34f515c1013305c92ab48dc2fb521664812c1955ff9492c9` |

The first plant attempt compared a supplied-operand graph with a computed-
operand graph, so its lack of propagation was not a valid causal control.  It
is neither deleted nor cited as passing: the retained
`association/keg_ulp_plant_inert_attempt.log` has SHA-256
`20bd2be209970c19e3bece7c8a9a25dd22e433c672a8d90b1787076cfa23e895`.
The corrected control compares identical override graphs with and without the
single changed word.

## Decision-45 year admission

The immutable member was produced cleanly at
`4d250301588d3ed0ad83fb20d6bf520e175d576e` with tag `year`, seed 0,
360 days, 2,160 steps, `dt=14,400 s`, fp64 and snapshots every six steps.  The
gate admitted every requested snapshot and its required T/S/u/v/ssh arrays,
then scored the fixed rows against `phase3/year_fromrest`:

| day | admitted before T3D wet RMS [K] |
|---:|---:|
| 30 | `6.89048490148956762e-5` |
| 60 | `1.93299736819368750e-4` |
| 90 | `1.86450211449135854e-3` |
| 120 | `1.05012568195104756e-3` |
| 180 | `3.58055101186670896e-3` |
| 240 | `1.64467419302924481e-2` |
| 300 | `1.35974041773198433e-2` |
| 360 | `1.12235739101672668e-2` |

All values confirm P1.  The landing report now registers every row before and
after; requires exact day set, producer commit and manifest contract; requires
month/year day-30 equality; and applies `after <= before` at days 240 and 360.
The command-level `year-day240-worse` plant changes only the candidate's day
240 value to the next larger fp64 value.  It exits nonzero and prints verbatim:

`STATUS PLANT-FIRED: year-day240-worse rejected by Decision-43/45 gate`

## Compiled program that actually ran

This round consumes the admitted Round-117 record and quotes its compiled
branch.  The live namelist selects vector C2 KEG at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/namelist_cfg:161-162`; the completed
record reports that selection at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/ocean.output:780-781`.  The compiled
initializer maps it to `np_VEC_c2` and permits the C2/Hollingsworth KEG choices
at `GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynadv.f90:339-345`.

The compiled stage program accumulates HPG, LDF, VOR, KEG and ZAD into the
same `Krhs`, in that order, at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:147-179`.
The selected C2 routine forms horizontal kinetic energy and performs the
in-place gradient writes at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:117-131`.
The first production-versus-isolated non-bit statement remains the U member
of the exact U/V write pair at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:129-130`.

The discriminator does not claim the compiled statement is wrong.  It asks
which complete-step JIT graph reproduces the separately materialized
WRITE-only boundary.  Because only insertion of the active mask
materialization closes the production boundary despite identical target
operand bits, the result is a compiler-graph classification, not authority to
add a mask or optimization barrier to shared production physics.

## Exact target and full-production arms

Round 119 reproduces before any new classification: production and isolated
are BIT through HPG/LDF/VOR, then one U cell differs at KEG/ZAD/ADV by
`4.1359030627651384e-25`; V remains BIT.  Ordinary versus NEMO-order final
counts remain 6,882/17,400 U and 6,566/17,100 V at
`8.470329472543003e-22`.  All five raw terms are BIT between production arms,
and each arm closes to its own live total.

The single target is wet:

| item | exact value |
|---|---|
| native U index | `[1,23,2]` |
| corresponding full-face index | `[1,24,2]` |
| active mask | `1.0` |
| after-VOR input | `-3.092914719338875e-9`, `0xbe2a9164327ae828` |
| unmasked addend | `-1.0819320780753281e-11`, `0xbda7cabc72fce380` |
| face-masked addend | same bits |
| WRITE-only masked addend | same bits |
| production after KEG | `-3.103734040119628e-9`, `0xbe2aa92eeeede50b` |
| isolated after KEG | `-3.1037340401196283e-9`, `0xbe2aa92eeeede50c` |
| production minus isolated | `+1` signed ULP |
| NEMO after KEG | `-3.103733663513917e-9`, `0xbe2aa92eb8a798ac` |

Authoritative production-JIT results:

| private one-variable arm | after-KEG U versus isolated | after-ZAD/ADV | versus baseline | complete returned state | disposition |
|---|---:|---:|---:|---:|---|
| unmasked materialized | 1 cell, `4.1359e-25` | same one cell | BIT | BIT | frozen prediction **REFUTED** |
| face-masked materialized | **BIT** | **BIT** | one cell, `4.1359e-25` | BIT | sole closing arm |
| literal subtract | 1 cell, `4.1359e-25` | same one cell | BIT | BIT | association not owner |

Every arm's HPG, LDF, VOR, KEG and ZAD inputs are BIT to baseline.  The
returned physical state is also BIT because this is a private WRITE-only
trace arm; no card consumes its alternate accumulator.  Under production
eager, baseline is already BIT to isolated and all three controls remain BIT.
Thus eager is retained as a labelled control and is not substituted for the
production result.

The corrected production KEG plant advances the targeted applied addend by
one ULP (`1.6155871338926322e-27 m s-2`).  HPG/LDF/VOR boundaries and all
unperturbed raw operands stay BIT; KEG/ZAD/ADV each move exactly the target
cell by `4.1359030627651384e-25`.  It exits nonzero and prints verbatim:

`ROUND120 ASSOCIATION-KEG-ULP STATUS PLANT-FIRED`

## Prediction disposition

| preregistered statement | verdict | evidence |
|---|---|---|
| immutable eight-row year table reproduces | CONFIRMED | all eight fp64 values exact |
| day-240 nextafter plant rejects and exits nonzero | CONFIRMED | command-level plant test |
| Round-119 one-cell production-JIT split reproduces | CONFIRMED | exact index, count and maximum |
| target is wet and unmasked/masked/WRITE addends have identical bits | CONFIRMED | mask `1.0`, common hex word |
| unmasked materialization closes | **REFUTED** | identical to baseline; one-cell split remains |
| only masked materialization closes | CONFIRMED | all KEG/ZAD/ADV boundaries BIT to isolated |
| literal subtraction closes | REFUTED | identical to baseline; one-cell split remains |
| production eager is BIT | CONFIRMED | all boundaries and arms BIT |
| result supplies a material landing candidate | REFUTED by scale and preregistration | one `4.136e-25` word; no production statement authorized |

## Landing stop, magnitude and blast radius

No production candidate exists, so Decision 43/45 is **NOT RUN as a
candidate comparison**.  The newly implemented year gate is exercised on its
immutable baseline and plants, not misrepresented as an after trajectory.
The production anchors therefore remain the inherited values:

| headline | unchanged inherited value |
|---|---:|
| kt2 T | `1.4210854715202004e-14 K` |
| kt2 S | `2.1316282072803006e-14 PSU` |
| kt2 U | `2.7377110452773967e-12 m s-1` |
| kt2 V | `3.2849219221489645e-12 m s-1` |
| kt3 T | `8.600419718618468e-7 K` |
| kt3 S | `6.979441735666114e-8 PSU` |
| day-30 T rms | `6.890484901489568e-5 K` |
| day-240 T rms | `1.6446741930292448e-2 K` |
| day-360 T rms | `1.1223573910167267e-2 K` |

The gate changes landing admission only.  The KEG arms are private
`_NEMOWSRK3TestHooks` tuples with a false default and cannot be selected by a
recipe.  GYRE, generic NEMO-GYRE, DINO, LOCK_EXCHANGE and OVERFLOW therefore
execute unchanged production.  DINO shares the underlying KEG/ZAD statements
and is **AT RISK** for any future production change; its cancellation makes a
GYRE-only inference invalid.  The tanks do not execute the private arm.  ORCA2
remains **UNMEASURED-WITH-SPEC**: record native operator and year entries, run
the same production-JIT/eager/plant classification, then its certified
trajectory and registered year rows before any shared landing.

The current measured magnitude order is unchanged at the stage boundary: W
is non-BIT in 18,000/18,000 cells at
`7.946658315637966e-7 m s-1`, and ZAD adds about `1.9e-9 m s-2`; the KEG
classification is one `4.1e-25` word.  But no controlled intervention has
measured W's day-240 contribution.  Decision 45 therefore forbids promoting
that instantaneous order into a year-owner ranking; the next round must make
that measurement.

## Verification

The Decision-43/45 gate suite, including the real CLI day-240 plant, ended:

- `8 passed in 9.62s`.

Two earlier development runs are retained rather than overwritten.  Before
the CLI subprocess test was added the suite ended `7 passed in 9.63s`.  The
first run after adding it correctly exercised the clean-stamp refusal while
that test file was still uncommitted and ended
`5 failed, 3 passed in 9.35s`; all five failures name the dirty producer.
After committing, the eight-test result above is the clean control.

The extended Round-83 discriminator suite ended:

- `12 passed in 0.50s`.

The mandatory combined full-tree attempt collected 8,211 tests and reached
98%, but ten xdist workers aborted during concurrent JAX compilation and the
worker pool disappeared.  It printed **no terminal summary line** and was
interrupted only after no pytest process remained.  Its partial log SHA-256
is `27bf2b945a5c919b2bd66cc183b5583139f664041e1e6455a1f5fc7e7220bacd`;
it is not called green.

All ten crash sites were replayed serially, expanding to 21 cases.  The
first command included two unqualified method IDs and ended literally
`no tests ran in 0.45s`; the IDs were corrected instead of hidden.  The
correct replay's literal terminal summary was:

- `4 failed, 17 passed in 397.01s (0:06:37)`.

All four failures are the pre-existing forward-mode-over-custom-VJP
parametrizations already present in the frozen known-red set.  The serial log
SHA-256 is
`7a7c288e8d731830a71c2a82baa3c7e7c20e23a9b5d8a8843e2c498d009fb88f`.

The frozen 87-node known-red list was also replayed directly at this tip.  Its
literal terminal summary was:

- `5 failed, 70 passed, 12 skipped, 8 warnings in 545.89s (0:09:05)`.

The five observed failures are a subset of the frozen list; `new versus
frozen` is empty.  The remaining 82 historical nodes now pass or skip at the
incoming tip, consistent with the intervening baseline-cleanup work.  The log
SHA-256 is
`54184cdb07bd3390e5eb1cf547a3f4c5d83788db294f6bf61090ec3e899f334e`.

Following the repository instruction to split compiler-heavy suites, the
first 69 fidelity files completed:

- `807 passed, 5 skipped in 160.89s (0:02:40)`.

The second 68-file xdist slice reached 87% with one failure marker but lost
its process before teardown and printed **no terminal summary line**.  Its
successively smaller retries are all retained: the first 17-file slice ended
`81 passed in 50.58s`; the next 17-file slice printed **no terminal summary
line**; its first eight files ended `86 passed in 29.26s`; its last nine files
again printed **no terminal summary line** under xdist.  The last nine were
then run serially: they collected 101 tests and reached the phase-3 stage
sweep at 39%; individual legacy sweep cases then took many silent minutes and
the round CPU bound was reached, so that command also printed
**no terminal summary line**.  None of these partial slices is called green
or used to erase the complete first-slice result.  Their SHA-256 values are,
in command order,
`1fe79b1e6b25619d958b02b1a870d447aa228ebc602382a6b49ec02711804d23`,
`bf3a52158d222c6bfb4b1e42414ec8cab9feeb3a56706463330e10789151222c`,
`b7d951b834021554101a47bcebd7556c55cb3496e227db704f10ed64627ebfc2`,
`5b4b8e40b0e765e18b01614d5b4c385497ba00af1aa55327bc26b28e885038ac`,
`28df1967c62cf93a5a9e8a4fc008bb565ed63cb234383ef0fb70785b9ef94c6b`,
`5f247cffc5b5ac5005261e8c6c88ce128c486e5d86f74ed1f89b54f1922dba0c`
and
`26f97dce1d6fd871a467693466a55f5baf4254e88bf3192a964f84921f0995b5`.

The first final focused run exposed two failures and ended literally:

- `2 failed, 122 passed, 1 warning in 347.34s (0:05:47)`.

The receipt-citation failure was caused by this round's diagnostic seams
rigidly moving already pinned local-source statements.  The map was moved by
exactly +2 lines in the model file and +4 lines in the tendency file with
every extent unchanged; all 16 affected historical receipt spans were moved
by the same displacement.  Commit
`738c3c852315762015bae99e255db664ffa0718b` records the map and commit
`5d015abbb0b2611cae6b89ac33d51f6d4e8509dd` records the receipt anchors.
The dedicated rerun then ended `16 passed in 1.97s`.

The second focused failure is unrelated to the changed latitude/longitude
path: the MPAS recipe's CG carry mixes float32 and float64.  Its isolated
candidate-tip run ended `1 failed, 1 warning in 2.79s`; the exact same node in
the untouched incoming-tip checkout ended
`1 failed, 1 warning in 3.04s`, with the same dtype exception.  It is thus
incoming pre-existing, though it postdates the frozen 87-node list.  The
final focused command explicitly deselected that one controlled node and
ended:

- `123 passed, 1 deselected in 339.75s (0:05:39)`.

The final focused-log SHA-256 is
`e5e26b7aa5bb7f3fcc425f91aff1eb57256e58a5dcde5f7c6907802939de2b73`;
the incoming-tip control is
`e6d6b3888e33627869b10c6df166f651ff05270c1d3a2c30584825a1729690d8`.

Finally, the clean-tree receipt citation gate found all six compiled-source
citations, no failures, no unmapped citations and no map-audit failures:
`status=PASS`.  The deliberately shifted KEG pair exited nonzero with
`status=FAIL` and `SYMBOL-NOT-AT-LINE`.  The final gate and plant JSON/log
SHA-256 values are
`b0da51a21495bba4be193fb42defa4ac83b33e4533faf356580396b193fa0297`
and
`5f88826d4cac121855414b1b78b01ea63d88e54bc7959ff88e09d84facb1c2ad`,
respectively.

## Independent review

The required separate command was run with `codex exec --sandbox read-only -C
/tmp/autopilot-work-5tqQh44i` and an adversarial prompt covering the complete
diff, compiled source, Decision-45 admission, exact one-variable arms, plant,
blast radius and no-trajectory verdict.  It could not initialize in the
mandated sandbox.  Its verbatim terminal verdict was:

`Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

Independent review is unavailable in-sandbox; no SHIP verdict is fabricated.
The preserved log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN — exact handoff to round 121

1. Return immediately to magnitude.  Do not optimize or land the one-cell KEG
   graph classification; it is fifteen-plus orders below the year gap and is
   not a source correction.
2. W is the current first/largest measured ZAD operand.  At its current-tip
   producer, preregister one controlled production-JIT intervention that
   replaces only W with the admitted NEMO stage value while leaving every
   other ZAD operand live.  Reproduce the 18,000-cell input row and the
   approximately `1.9e-9` ZAD increment first; include eager only as a
   separately labelled control and require a production plant.
3. Measure the intervention through the full kt1..10 ladder, days 1--30 and a
   360-day from-rest run.  Rank it by the registered day-240 effect first,
   then day 360 and day 30.  This is attribution, not yet permission to carry
   an oracle field: any real candidate must name W's first non-bit producer
   from compiled source and independently satisfy Decision 43/45.
4. If the controlled W intervention has no material day-240 effect, record the
   refutation and return to the year-owner decomposition rather than chasing
   another last-bit stage row.  If it is material, walk W's producer in
   compiled order and preregister the first production-JIT non-bit statement.
5. Any shared W/ZAD production candidate measures DINO and all recipe-derived
   executing cards before landing.  Keep ORCA2 `UNMEASURED-WITH-SPEC` until
   its native record and certified year trajectory exist.

No NEMO acquisition and no user configuration or carried-state decision are
needed for this handoff.
