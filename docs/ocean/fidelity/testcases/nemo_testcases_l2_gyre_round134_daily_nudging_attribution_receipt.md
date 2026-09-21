# NEMO testcase L2 GYRE phase 3 — round 134 daily-nudging attribution receipt

Date: 2026-09-21

Incoming tip: `91864f1d9e9c68f2c44ea7be918b8936ded8a4a1`

Status: **HELD — daily tracer-state reset removes 99.6832% of the day-240
temperature RMS gap and is the measured owner family, refuting the frozen
vector-family prediction; this is intervention leverage, not yet a physical
statement, so no physics or configuration lands.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round134/`

## Outcome first

The acquired Round-132/133 record is admitted. It has exactly 360 daily
completed-step boundaries, every required one of 18 fields, and all twelve
monthly overlaps are bit-identical to the existing NEMO year record. A fresh
unnudged control at the Round-134 measurement commit reproduces the immutable
Round-130 legoESM arm bit-for-bit for `T`, `S`, `u`, `v`, and `ssh` at all 360
days.

Four independent production-JIT year runs then reset one state family to
NEMO after each day's pre-reset snapshot. At day 240 the free-run T3D RMS gap
is `1.6446740233175394e-2 K`:

| rank | reset family | reset T3D RMS (K) | removed T3D RMS (K) | fraction removed |
|---:|---|---:|---:|---:|
| 1 | tracer `tn/sn` | `5.209685836156627e-5` | `1.6394643374813826e-2` | `99.683239%` |
| 2 | vector + velocity histories | `1.537035134357157e-2` | `1.076388889603824e-3` | `6.544694%` |
| 3 | TKE state/coefficient memory | `1.6325689194579378e-2` | `1.2105103859601576e-4` | `0.736018%` |
| 4 | SSH + surface histories | `1.6445428646963186e-2` | `1.311586212207616e-6` | `0.007975%` |

Thus the preregistered claim that vector/history reset would win is
**REFUTED**. Tracer reset is 15.23 times more effective than vector reset at
day 240. This experiment identifies the state family whose repeated daily
replacement controls the accumulated gap. It does **not** identify the first
wrong tracer-producing statement: directly pinning `T/S` can erase errors
from any upstream process that reaches tracers. No statement, operator, or
scheme is claimed fixed by this result.

The first tracer-arm temperature difference from the free arm is at day 2,
as preregistered: 21,119 unequal cells and `6.128821775019916e-7 K` wet-cell
RMS. Its strongest depth band is 0–100 m (63.3398% of summed `dT^2`,
`9.445628492641942e-7 K` RMS) and its strongest longitude third is the western
third (48.4170%, `7.386463459394441e-7 K` RMS). Those two location predictions
are **CONFIRMED**. The maximum signed difference is
`-9.908159242399961e-6 K` at `(j=2, i=30, k=4)`. This is the birthplace of the
*reset response*, not yet proof that the underlying first non-bit statement
is located there.

## Compiled-source contract and boundary timing

The cited source is the compiled branch of the acquired record, not generic
NEMO source. NEMO executes the first external solve and stage 1 at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:188-201`. At the end
of the accepted RK3 step it swaps the new state into `Nbb` and constructs the
next-step extrapolated SSH in `Naa` at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:222-226`, then invokes
the restart writer after diagnostics at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:249-260`.

The compiled writer stores `sshn`, `un`, `vn`, `tn`, `sn`, `uu_n`, `vv_n`,
and `ssha` at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:176-184`. The active
time-split branch stores the six barotropic histories at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/dynspg_ts.f90:974-980`, and the
active TKE branch stores `en`, `avt_k`, `avm_k`, and `dissl` at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/zdftke.f90:895-901`.

The otherwise easy-to-lose `ssha` distinction is compiled behavior. Restart
loading reads it into the RK3 `Kaa` slot (with an explicit old-format fallback)
at `GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:354-370`. At the
next step the external solve converts that `Kaa` SSH to the free-surface ratio
and calls W/ZAD at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stp2d.f90:150-166`; W consumes
the difference between the `Kaa` and `Kbb` ratios at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/sshwzv.f90:293-299`. The private
diagnostic override therefore supplies the recorded `ssha` only to the next
production step and defaults to `None`; it does not add carried state or
change a normal run.

The harness snapshots before resetting at each completed daily boundary and
the replacement is first consumed on the following step. All four arms are
therefore bit-identical to the free control on day 1 for all five scored
fields. Each intervention applies at 359 boundaries, days 1–359; day 360 is
scored and is not followed by an unused reset.

## Frozen prediction ledger

* **P0 CONFIRMED.** The clean stamped audit found exactly steps
  `6,12,...,2160`, 360/360 files, 18/18 required variables, the requested
  cadence, and 12/12 bit-identical monthly overlaps. The final admitted audit
  is `daily_record_audit_stamped.json`, SHA-256
  `4da31077628cac2e6a4fb973212654fca635fdbb56389e0e8180d0da0ce6a787`.
* **P1 CONFIRMED.** The unnudged control is bit-identical to the immutable arm
  on all 360 days and all five fields. Each reset arm declares exactly its
  registered variables, uses the production `_step_jitted` route, applies 359
  post-snapshot resets, and has zero day-1 unequal cells.
* **P2 PARTLY REFUTED.** The predicted vector winner is **REFUTED** by the
  tracer arm's 15.23-fold larger day-240 removal. The predicted day-2 birth,
  upper-100-m band, and western third are **CONFIRMED** for the actual winner.
* **P3 CONFIRMED.** No physics, card, production carried state, NEMO source,
  or configuration changed. Only a default-off diagnostic seam and the
  existing year/scoring instruments were extended. Status remains HELD.
* The three-step cadence is **UNMEASURED** because the oracle record is daily
  only. No daily value was interpolated or reused as a three-step boundary.

## Complete registered score table

Each cell is `reset-vs-NEMO RMS (free-minus-reset RMS)`. Positive parentheses
mean the reset reduced that field's gap; negative values mean it worsened.
These 32 table rows contain all 160 preregistered family/day/field rows. The
machine-readable registry is `daily_reset_attribution.json`, SHA-256
`0702a2bbc114a27c66eeb3b1555e731ce06d1989247d7afc27318f909e7c57cd`;
it reports `registered_row_count=160` and
`all_moved_rows_registered=true`.

### Tracer reset

| day | T (K) | S (g/kg) | u (m/s) | v (m/s) | ssh (m) |
|---:|---:|---:|---:|---:|---:|
| 30 | `5.860185726268e-06 (+6.304412915199e-05)` | `6.200055783841e-07 (+1.116121690246e-05)` | `6.440563554702e-07 (+4.936041670447e-06)` | `6.539416449357e-07 (+3.966136446212e-06)` | `7.341184477733e-07 (+6.068663314293e-06)` |
| 60 | `1.224663580644e-05 (+1.810532033976e-04)` | `2.691950182293e-06 (+1.743136947625e-05)` | `9.139026765848e-07 (+1.959096442540e-05)` | `8.174672033305e-07 (+7.560477703729e-05)` | `7.183357760428e-07 (+1.608424291760e-05)` |
| 90 | `4.752415644962e-05 (+1.816977566869e-03)` | `1.365152881965e-05 (+2.211340750241e-04)` | `8.425716901307e-05 (+7.414366108179e-05)` | `1.155481188090e-04 (+5.545646115865e-05)` | `1.077655888848e-06 (+3.432963332049e-05)` |
| 120 | `1.977910658916e-04 (+8.523338106879e-04)` | `6.229632717359e-05 (+1.804505395860e-04)` | `1.031545018319e-04 (+2.832662723153e-05)` | `1.352150560057e-04 (+1.282108435494e-04)` | `1.122853641338e-06 (+4.470387523193e-05)` |
| 180 | `1.064116454204e-04 (+3.474138647869e-03)` | `2.847706437372e-05 (+8.716302460929e-04)` | `5.609548923103e-05 (+7.123157250416e-05)` | `7.779568181617e-05 (+9.212247836806e-05)` | `1.378987460510e-06 (+7.719326853257e-05)` |
| 240 | `5.209685836157e-05 (+1.639464337481e-02)` | `7.787127923652e-06 (+1.223932731015e-03)` | `2.774939790067e-05 (+2.795451738693e-04)` | `3.619015721847e-05 (+3.999049049748e-04)` | `1.703709828694e-06 (+1.917694626610e-04)` |
| 300 | `1.926426582793e-04 (+1.340475977827e-02)` | `1.958287969829e-05 (+1.190507767702e-03)` | `4.446261661436e-05 (+2.710703512830e-04)` | `1.028151791506e-04 (+1.931069450639e-04)` | `1.989335496581e-06 (+1.767574259876e-04)` |
| 360 | `1.505305862752e-04 (+1.107304066157e-02)` | `2.823024503989e-05 (+1.088964063498e-03)` | `1.021786390408e-05 (+4.248163691465e-04)` | `2.499508202488e-06 (+4.234235788991e-04)` | `2.063490064665e-06 (+1.910367897239e-04)` |

### Vector/history reset

| day | T (K) | S (g/kg) | u (m/s) | v (m/s) | ssh (m) |
|---:|---:|---:|---:|---:|---:|
| 30 | `5.518160682673e-05 (+1.372270805153e-05)` | `1.095438016077e-05 (+8.268423200764e-07)` | `3.907918521484e-06 (+1.672179504432e-06)` | `3.738825538230e-06 (+8.812525529170e-07)` | `4.520499114313e-06 (+2.282282647754e-06)` |
| 60 | `1.849474677979e-04 (+8.352371406151e-06)` | `1.794113818526e-05 (+2.182181473282e-06)` | `3.521334322493e-05 (-1.470847612294e-05)` | `1.224413269256e-04 (-4.601908268498e-05)` | `1.076399959211e-05 (+6.038579101535e-06)` |
| 90 | `1.814905330946e-03 (+4.959639237291e-05)` | `2.163969371768e-04 (+1.838866666702e-05)` | `1.158215136634e-04 (+4.257931643144e-05)` | `1.476196192877e-04 (+2.338496067997e-05)` | `2.542558036776e-05 (+9.981708841579e-06)` |
| 120 | `9.923333756461e-04 (+5.779150093339e-05)` | `2.398860085230e-04 (+2.860858236651e-06)` | `1.077233188612e-04 (+2.375781020220e-05)` | `1.265159634777e-04 (+1.369099360774e-04)` | `2.753731141072e-05 (+1.828941746254e-05)` |
| 180 | `3.510398874624e-03 (+7.015141866582e-05)` | `8.953373936660e-04 (+4.769916800632e-06)` | `5.677937050178e-05 (+7.054769123341e-05)` | `7.292834313381e-05 (+9.698981705042e-05)` | `4.567534340193e-05 (+3.289691259115e-05)` |
| 240 | `1.537035134357e-02 (+1.076388889604e-03)` | `1.228257375863e-03 (+3.462483076045e-06)` | `2.922677367934e-04 (+1.502683497658e-05)` | `4.028763371150e-04 (+3.321872507830e-05)` | `1.378482434922e-04 (+5.562492899755e-05)` |
| 300 | `1.314725712900e-02 (+4.501453075445e-04)` | `1.199721711745e-03 (+1.036893565590e-05)` | `3.085681388253e-04 (+6.964829072047e-06)` | `2.081048630297e-04 (+8.781726118475e-05)` | `1.183791557219e-04 (+6.036760576226e-05)` |
| 360 | `1.066408930617e-02 (+5.594819416715e-04)` | `1.066015638000e-03 (+5.117867053767e-05)` | `3.963069406308e-04 (+3.872729241977e-05)` | `2.672642650274e-04 (+1.586588220742e-04)` | `1.185653146311e-04 (+7.453496515745e-05)` |

### SSH/history reset

| day | T (K) | S (g/kg) | u (m/s) | v (m/s) | ssh (m) |
|---:|---:|---:|---:|---:|---:|
| 30 | `6.890478239662e-05 (-4.675183559109e-10)` | `1.178062281645e-05 (+5.996643935519e-10)` | `5.569985727414e-06 (+1.011229850301e-08)` | `4.588448453436e-06 (+3.162963771185e-08)` | `7.090934242124e-06 (-2.881524800575e-07)` |
| 60 | `1.932958155833e-04 (+4.023620786596e-09)` | `2.012356927963e-05 (-2.496210902199e-10)` | `2.048671193512e-05 (+1.815516686005e-08)` | `7.640962599966e-05 (+1.261824095821e-08)` | `1.851329796141e-05 (-1.710719267772e-06)` |
| 90 | `1.864365855267e-03 (+1.358680518614e-07)` | `2.347981267327e-04 (-1.252288891535e-08)` | `1.583986322719e-04 (+2.197822936340e-09)` | `1.709830740581e-04 (+2.150590956902e-08)` | `3.839223146221e-05 (-2.984942252873e-06)` |
| 120 | `1.049538130852e-03 (+5.867457279361e-07)` | `2.426487531735e-04 (+9.811358615090e-08)` | `1.314581459725e-04 (+2.298309090582e-08)` | `2.634069405318e-04 (+1.895902333000e-08)` | `5.262425179905e-05 (-6.797522925786e-06)` |
| 180 | `3.585011030057e-03 (-4.460736767029e-06)` | `9.018197120946e-04 (-1.712401627954e-06)` | `1.273091194956e-04 (+1.794223956900e-08)` | `1.692762755411e-04 (+6.418846431161e-07)` | `9.725242877374e-05 (-1.868017278066e-05)` |
| 240 | `1.644542864696e-02 (+1.311586212208e-06)` | `1.230863473528e-03 (+8.563854107862e-07)` | `3.073384311699e-04 (-4.385939993156e-08)` | `4.359836100090e-04 (+1.114521843037e-07)` | `2.197268132521e-04 (-2.625364076238e-05)` |
| 300 | `1.359580109271e-02 (+1.601343833060e-06)` | `1.208918161647e-03 (+1.172485753408e-06)` | `3.154011598952e-04 (+1.318080021509e-07)` | `2.956836712291e-04 (+2.384529853375e-07)` | `2.239256974432e-04 (-4.517893595898e-05)` |
| 360 | `1.125014725808e-02 (-2.657601023997e-05)` | `1.147073538834e-03 (-2.987923029573e-05)` | `4.379337784974e-04 (-2.899545446774e-06)` | `4.316140899323e-04 (-5.691002830681e-06)` | `2.531437797144e-04 (-6.004349992585e-05)` |

### TKE reset

| day | T (K) | S (g/kg) | u (m/s) | v (m/s) | ssh (m) |
|---:|---:|---:|---:|---:|---:|
| 30 | `6.883505999215e-05 (+6.925488611054e-08)` | `1.178009064758e-05 (+1.131833259935e-09)` | `5.577380403411e-06 (+2.717622506020e-09)` | `4.617667813040e-06 (+2.410278107609e-09)` | `6.801669054145e-06 (+1.112707921259e-09)` |
| 60 | `2.096059445452e-04 (-1.630610534112e-05)` | `2.063210297474e-05 (-5.087833162052e-07)` | `1.895109658288e-05 (+1.553770519103e-06)` | `1.614714861095e-05 (+6.027509562967e-05)` | `1.678072098880e-05 (+2.185770483717e-08)` |
| 90 | `1.603234120877e-03 (+2.612676024422e-04)` | `2.439059604066e-04 (-9.120356562840e-06)` | `1.627566028248e-04 (-4.355772729941e-06)` | `1.726881796564e-04 (-1.683599688679e-06)` | `3.337916432801e-05 (+2.028124881327e-06)` |
| 120 | `9.242011254319e-04 (+1.259237511476e-04)` | `2.144700672427e-04 (+2.827679951689e-05)` | `2.365470147450e-04 (-1.050658856816e-04)` | `3.038048690067e-04 (-4.037896945162e-05)` | `4.550089640913e-05 (+3.258324641337e-07)` |
| 180 | `3.555433867160e-03 (+2.511642612960e-05)` | `8.945891246026e-04 (+5.518185864084e-06)` | `1.095583063965e-04 (+1.776875533870e-05)` | `1.544985681177e-04 (+1.541959206652e-05)` | `7.847840533428e-05 (+9.385065879541e-08)` |
| 240 | `1.632568919458e-02 (+1.210510385960e-04)` | `1.228208967320e-03 (+3.510891619141e-06)` | `2.726909157323e-04 (+3.460365603766e-05)` | `3.863951564603e-04 (+4.969990573297e-05)` | `1.705323464548e-04 (+2.294082603497e-05)` |
| 300 | `1.384030592907e-02 (-2.429034925211e-04)` | `1.216635122375e-03 (-6.544474974344e-06)` | `3.073614521842e-04 (+8.171515713124e-06)` | `2.264135100000e-04 (+6.950861421441e-05)` | `1.734528574582e-04 (+5.293904026015e-06)` |
| 360 | `1.060857605191e-02 (+6.149951959368e-04)` | `1.111543673622e-03 (+5.650634915734e-06)` | `4.172282805266e-04 (+1.780595252402e-05)` | `3.422486825464e-04 (+8.367440455514e-05)` | `1.931653669606e-04 (-6.508717201390e-08)` |

## Controls and provenance

The first admission attempt terminated before a verdict because its error
formatting indexed an empty mismatch list even on the clean path. That is a
gate defect, not a record failure; it is retained in
`daily_record_audit.log`. The repaired clean run admitted the record, and the
final worktree-stamped rerun admitted the same 360 files and values. No arm
used data from the failed attempt.

The acquisition has a 360-file SHA-256 manifest and a final ready marker.
Every year-arm manifest records the clean measurement commit, fp64/libm
policy, source audit digest, family registry, exact list and digest of all 359
source boundaries, and wall time. Manifest digests are:

The four interventions and the control were measured at clean commit
`078e3e3c8b7c875466e6edab052de140864f779d`. After the reporter-stamp
ratchet was repaired, the unchanged record was readmitted at clean commit
`01c5fa2752862efc43dd64b8b6e2bb013d18d0ae`; the final stamped audit has the
same inventory, schema, and twelve bit-identical monthly twins as the audit
consumed by the arms.

| arm | manifest SHA-256 | wall time |
|---|---|---:|
| control | `9375124f0f32beb882cbfe2d4ad518d9c1bb0a43d368c5390237c9f96f89c82a` | `1748.13 s` |
| tracer | `37d86f29a3a291890da0a97aa86a6406ae82038acba6fd8eb802bc275f79ccf8` | `1820.25 s` |
| vector | `ee4fe33d70a7df1eb3066b4f02b7609ada9564920b363167c3d9f4560f1d2b55` | `1837.52 s` |
| SSH | `433b28bd45d23ad5e1cecbcd5c348804e4a720e9e58ef93e4f753d2a383b1d00` | `1892.63 s` |
| TKE | `f7a9258decd6559ae89bb896c25d3983356bde5d1b6b74f245ddda4cbbbb5a1b` | `1840.61 s` |

All fail-closed controls were run against otherwise valid inputs:

| control | exit | decisive output |
|---|---:|---|
| missing daily boundary | 1 | `STATUS PLANT-FIRED: missing-boundary` |
| missing required variable | 1 | `STATUS PLANT-FIRED: required-variable` |
| one-ULP monthly overlap | 1 | `STATUS PLANT-FIRED: monthly-overlap-ulp` |
| one-ULP daily reset source | 1 | `STATUS PLANT-FIRED: daily-source-ulp` |
| family registry mutation | 1 | `STATUS PLANT-FIRED: daily-family-registry` |
| 159/160 score rows | 1 | `STATUS PLANT-FIRED: daily-attribution-registry` |
| unplanted record self-check | 0 | `STATUS PASS` |
| unplanted reset self-check | 0 | `STATUS PASS` |

The production seam test proves three separate properties through the actual
JIT step: omitted override and explicit `None` are bit-identical, a distinct
valid override moves the result, and a one-ULP source mutation is rejected by
the harness before it can be consumed. The score-registry plant drops one
specific valid row; it cannot be satisfied by merely asserting that some row
moved.

## Trajectory, landing, and cross-card disposition

This is a diagnostic intervention, not a candidate implementation. It has no
meaningful Rule-12 or Decision-43/45 landing comparison: all normal runs keep
the override at `None`, and the unnudged year control is bit-identical to the
immutable arm. The GYRE ladder, `kt2 U/V`, `kt3 T/S`, month score, day-240
before arm, and day-360 before arm therefore do not change. No new immutable
after arm is created.

| campaign surface | result |
|---|---|
| GYRE normal trajectory | 360 days, all five fields bit-identical to before |
| DINO | default-off seam; no card or normal trajectory change |
| LOCK_EXCHANGE / OVERFLOW / tanks | statement does not execute unless a private test override is supplied |
| ORCA2 | UNMEASURED-WITH-SPEC; default-off seam and card untouched |
| physics/configuration/carried state | none landed |

DINO shares the underlying W/ZAD implementation, but this round changes no
selected statement in that implementation: the extra operand is private,
optional, and absent in every card run. A DINO numeric before/after is thus
not claimed; its normal branch is covered by the default-off identity and
existing DINO focused tests. This does not turn an unmeasured intervention on
DINO into a measured fidelity row.

## Independent adversarial review

The required separate command was invoked from clean draft-receipt commit
`c9a8f6cf5` with `codex exec --sandbox read-only -C` and a prompt that tried to
refute record completeness, reset timing, family isolation, production-JIT
execution, `ssha` routing, all 160 score rows, the ownership caveat, plants,
cross-card scope, and the incomplete broad tests. It returned exit 1 before
reading the diff. Its complete Codex output, verbatim, was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Thus **independent review unavailable in-sandbox**. The retained log is
`phase3/round134/codex_review.log`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
It issued no verdict and therefore no `DO NOT SHIP` verdict. This receipt does
not mislabel the infrastructure failure as an approving review.

## Verification

All commands used the required CPU/fp64 environment. The final focused
production-path invocation reported exactly:

```text
40 passed in 245.90s (0:04:05)
```

It covered the Round-134 tests, the Round-131 record gate, coupled W/ZAD, and
the full production MLF/JIT transcription tests. The earlier focused
instrument run reported `31 passed in 169.21s (0:02:49)`. After repairing the
two new fail-closed metadata regressions, their direct regression set reported
`27 passed in 14.24s`. The final pre-receipt citation suite reported
`16 passed in 1.95s`.

The requested all-tree run was attempted, not silently omitted. One combined
`tests/ocean/fidelity tests/ocean/unit -n 12` invocation collected 8,288 tests,
reached 71%, then a JAX compiler worker died in
`backend_compile_and_load`; a replacement worker did not finish within the
round budget. It has no pytest summary line and is **INCOMPLETE**, so it is not
represented as a pass. A separate fidelity-tree run collected 1,444 tests and
reached 94% without completing. Splitting that tree produced these exact
completed summaries:

```text
364 passed, 2 skipped in 132.29s (0:02:12)
1 failed, 356 passed, 4 skipped in 51.46s
121 passed in 234.65s (0:03:54)
```

The single failure in the middle line was the worktree-stamp static ratchet;
it detected this round's new reporter. It was fixed, and the final
`27 passed` regression run includes that ratchet. One earlier isolated
Round-129 control failed only because extending the shared year harness moved
its pinned harness digest; the pin was updated only after the 360-day,
five-field bit-identical control reproduced the certified trajectory, and it
also passes in the final `27 passed` set. Therefore the final completed
affected suites add no failure ID to the 87-ID pre-existing-red list. Because
the two whole-tree invocations did not reach summaries, a complete final
87-ID set comparison is **UNMEASURED**, not claimed green.

The many completed file-level fidelity shards are indexed by their literal
pytest summary lines in `phase3/round134/fidelity_file_*.log`; the exact
per-invocation ledger follows. Lines marked INCOMPLETE are deliberately not
converted into pass claims.

| invocation log | exact pytest summary or terminal state |
|---|---|
| `citation_suite_pre_receipt.log` | `3 failed, 13 passed in 1.95s` |
| `citation_suite_reanchored.log` | `2 failed, 14 passed in 1.84s` |
| `citation_suite_reanchored_clean.log` | `1 failed, 15 passed in 1.91s` |
| `citation_suite_reanchored_clean_v2.log` | `16 passed in 1.95s` |
| `citation_suite_final.log` | `16 passed in 1.94s` |
| `focused_tests.log` | `31 passed in 169.21s (0:02:49)` |
| `focused_production_tests.log` | `40 passed in 245.90s (0:04:05)` |
| `fail_closed_regression_tests.log` | `27 passed in 14.24s` |
| `fidelity_chunk_0.log` | `364 passed, 2 skipped in 132.29s (0:02:12)` |
| `fidelity_chunk_1.log` | `1 failed, 356 passed, 4 skipped in 51.46s` (repaired and covered by final 27-pass run) |
| `fidelity_p2_s0.log` | `121 passed in 234.65s (0:03:54)` |
| `fidelity_file_test_baro_fixed_bias_wall_map.log` | `79 passed in 0.20s` |
| `fidelity_file_test_compare.log` | `14 passed in 0.35s` |
| `fidelity_file_test_coriolis_omega_routing_audit.log` | `10 passed in 0.27s` |
| `fidelity_file_test_fidelity_card_constructibility.log` | `16 passed in 14.77s` |
| `fidelity_file_test_fixtures_validate.log` | `10 passed in 0.27s` |
| `fidelity_file_test_gyre_round107_tke_rhs_intermediate_walk.log` | `9 passed in 1.69s` |
| `fidelity_file_test_gyre_round108_bn2_intermediate_walk.log` | `15 passed in 1.00s` |
| `fidelity_file_test_metrics_tier2.log` | `5 passed in 0.27s` |
| `fidelity_file_test_metrics_tier5.log` | `4 deselected in 0.26s` |
| `fidelity_file_test_mitgcm_advection_gyre_recipe.log` | `7 passed, 1 skipped in 1.96s` |
| `fidelity_file_test_mitgcm_front_relax_recipe.log` | `5 passed, 1 deselected in 1.33s` |
| `fidelity_file_test_mitgcm_gyre_canonical.log` | `3 passed, 2 deselected in 54.39s` |
| `fidelity_file_test_mitgcm_monitor.log` | `11 passed in 0.32s` |
| `fidelity_file_test_mitgcm_recipe_card.log` | `4 passed in 4.13s` |
| `fidelity_file_test_mlf_step_mechanism_ab.log` | `5 passed in 0.81s` |
| `fidelity_file_test_nemo_dino_mesh.log` | `16 passed in 48.39s` |
| `fidelity_file_test_nemo_dino_step1_gate.log` | `5 passed in 11.74s` |
| `fidelity_file_test_nemo_testcase_full_statistics.log` | `15 passed in 0.31s` |
| `fidelity_file_test_nemo_testcase_l1_tanks_round33_zdf_rule12.log` | `15 passed in 7.46s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_oracle_census.log` | `4 passed in 0.05s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_phase3_gate.log` | `23 passed in 4.21s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round129_spread_floor_gate.log` | `1 failed, 7 passed in 10.58s` (stale digest repaired; final 27-pass run green) |
| `fidelity_file_test_nemo_testcase_l2_gyre_round29_zdf_matrix.log` | `9 passed in 2.01s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round32_ordering.log` | `22 passed in 3.49s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round33_stage_arm.log` | `22 passed in 9.04s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round42_ldfslp.log` | `3 passed in 7.28s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round45_ablation_gate.log` | `2 passed in 0.37s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round51_live_operands.log` | `3 passed in 0.42s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round54_tracer_decomposition.log` | `5 passed in 0.05s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round63_krhs_split.log` | `8 passed in 0.64s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round71_fct_stage2_gate.log` | `4 passed in 0.30s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round72_stage1_transport_gate.log` | `2 passed in 0.28s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round73_stage1_transport.log` | `3 passed in 0.29s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round76_advmean_walk.log` | `4 passed in 0.29s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round82_btstep_walk.log` | `6 passed in 0.64s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round85_bundle_gate.log` | `3 passed in 0.46s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_round86_zad_operands.log` | `3 passed in 0.29s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_year_fromrest.log` | `30 passed in 17.86s` |
| `fidelity_file_test_nemo_testcase_l2_gyre_year_owners.log` | `27 passed in 30.31s` |
| `fidelity_file_test_nemo_testcase_l4_orca2_hpg_model_arm_probe.log` | `3 passed in 0.28s` |
| `fidelity_file_test_nemo_testcase_lock_slow_forcing_owner.log` | `3 passed in 0.75s` |
| `fidelity_file_test_nemo_testcase_phase2_gate.log` | `10 passed in 8.38s` |
| `fidelity_file_test_nemo_testcase_phase3_trajectory_gate.log` | `9 passed in 0.30s` |
| `fidelity_file_test_nemo_testcase_round34_tank_zdf_removal.log` | `10 passed in 7.48s` |
| `fidelity_file_test_nemo_testcase_round35_stamp_scope.log` | `8 passed in 0.46s` |
| `fidelity_file_test_nemo_testcase_round39_k33_placement.log` | `8 passed in 58.04s` |
| `fidelity_file_test_nemo_testcase_rule12_hpg_eligibility.log` | `7 passed in 0.32s` |
| `fidelity_file_test_oceananigans_bickley_config.log` | `3 passed in 3.61s` |
| `fidelity_file_test_oceananigans_gyre_wind_convention.log` | `3 passed in 1.46s` |
| `fidelity_file_test_oceananigans_internal_tide.log` | `3 passed in 3.14s` |
| `fidelity_file_test_oceananigans_recipe_card.log` | `7 passed in 0.27s` |
| `fidelity_file_test_oceananigans_runner.log` | `6 passed in 1.11s` |
| `fidelity_file_test_oceananigans_tendency_align.log` | `2 passed in 0.30s` |
| `fidelity_file_test_precedence.log` | `11 passed in 0.27s` |
| `fidelity_file_test_registry.log` | `10 passed in 0.26s` |
| `fidelity_file_test_run_acc_freerun.log` | `2 passed, 1 deselected in 2.38s` |
| `fidelity_file_test_run_comparison_script.log` | `11 passed in 0.29s` |
| `fidelity_file_test_veros_global_freerun.log` | `5 passed in 0.33s` |
| `fidelity_file_test_veros_runner.log` | `23 passed, 7 deselected in 1.80s` |
| `full_ocean_tests.log` | **INCOMPLETE:** 8,288 collected, worker crashed, last progress 71%, no summary |
| `full_fidelity_tests.log` | **INCOMPLETE:** 1,444 collected, last progress 94%, no summary |
| `fidelity_chunk_2.log` | **INCOMPLETE:** 337 collected, progress reached 100%, teardown never produced summary |
| `fidelity_p2_s1.log` | **INCOMPLETE:** 59 collected, 50 dots, no summary |
| `fidelity_p2_s1_without_round134.log` | **INCOMPLETE:** 53 collected, 44 dots, no summary |
| `unit_chunk_0.log` | **INCOMPLETE:** 140 collected, last progress 51%, no summary |
| `fidelity_file_test_nemo_testcase_phase3_stage_sweep_gate.log` | **INCOMPLETE:** 10 collected, first 7 passed, long historical integration stopped, no summary |

The final unplanted citation run found nine citations, zero unmapped
citations, zero failures, an empty whole-map audit, all nine internal controls
firing, and `status: PASS`. Shifting the registered external-step citation by
two lines returned exit 1 with `status: FAIL` and `SYMBOL-NOT-AT-LINE`. The
retained artifacts are `phase3/round134/citation_gate.json` and
`phase3/round134/citation_gate_shifted_plant.json`.

## OPEN — next round

1. Stay with the measured tracer family. Preregister and run two independent
   daily interventions from the same admitted record: `T`-only (`tn`) and
   `S`-only (`sn`). The frozen next discriminator should ask which one carries
   the day-240 T3D removal; do not call either a statement before measurement.
2. For the dominant subfamily, reduce cadence only at boundaries the record
   actually contains (for example every 2, 4, or 8 days). The requested
   three-step cadence remains unavailable and must not be synthesized.
3. Locate the first daily interval in which the dominant subfamily's reset
   prevents the western-upper-100-m growth, then use existing per-process
   tracer traces and the compiled stage order to substitute one source family
   at a time. The deliverable is the first non-bit consumed operand/statement,
   cited from the compiled branch, not another circular process label.
4. Keep the vector, TKE, and SSH families ranked as measured controls. Do not
   discard their signed worsenings or reinterpret the vector refutation.
5. Re-run the full fidelity/unit trees only after clearing the process-level
   JAX compiler cache between shards; the current incomplete attempts cannot
   certify the whole tree.

No NEMO acquisition and no user configuration decision are needed for this
next discriminator. Production remains on the incoming physical trajectory.
