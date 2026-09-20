# NEMO testcase L2 GYRE phase 3 — round 130 year-scored patch receipt

Date: 2026-09-20

Incoming tip: `a18ba326ec94ab4afbac28de681e69e714621ec8`

Status: **LANDED — the held round-109 vector stage-1 handoff is the only
single arm that passes the registered month, ladder, day-240, and day-360
criteria.  It decreases day-240 T3D RMS by
`1.6971170545276859e-9 K`; this is a real but very small
`1.0318864743672114e-7` fraction of the remaining gap.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round130/`

## Outcome first

Eight preserved, locally source-exact patches were applied one at a time to
the same incoming tip and each was run independently for 360 days.  The
round-109 stage-1 vector handoff ranks first by the frozen day-240 ordering
and is the only Decision-43/45 pass.  Relative to the independently reproduced
incoming-tip control, it changes the three required T3D RMS headlines as
follows:

| day | before (K) | after (K) | after - before (K) |
|---:|---:|---:|---:|
| 30 | `6.89048490148956762e-05` | `6.89043148782590898e-05` | `-5.34136636586328081e-10` |
| 240 | `1.64467419302924481e-02` | `1.64467402331753935e-02` | `-1.69711705452768591e-09` |
| 360 | `1.12235739101672668e-02` | `1.12235712478436639e-02` | `-2.66232360289497816e-09` |

The landing does **not** close the year gap.  Its day-240 fractional reduction
is about one part in ten million.  It lands because it is NEMO's compiled
statement and clears every frozen criterion, not because this tiny response
is relabelled as the macroscopic owner.

The canonical oracle-relative comparison still says `FAIL`: Decision 43
explicitly supersedes the older no-movement clause for kt>=2, and this
candidate moves both improving and worsening cells.  The Decision-43/45 gate,
not that legacy verdict string, is the landing authority.

## Compiled-source statements

The eight local proofs were re-opened against the compiled record branches.
No configuration selector, stabilizer, or uncompiled source arm is used.

* The round-62 coefficient/content candidate transcribes the matrix
  coefficients at
  `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:461-477`, the first
  recurrence at
  `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:523-528`, the complete
  thickness-weighted content RHS at
  `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:545-564`, and the
  back-substitution at
  `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:573-578`.
* The round-88/89/97/99 family follows the external solve and three stage
  calls at
  `GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:188-201`, the
  completed-step swap and W setup at
  `GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:220-226`, the W
  recurrence at
  `GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300`, and the
  vector RK assignment at
  `GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:671-674`.
* The round-105 routing split follows the step-entry time-level call at
  `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:167-168`, its binding
  into `zdf_sh2` at
  `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:319-320`, and the two
  shear divisors at
  `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:97-109`.
* The round-112 FCT candidate uses the metric transports written at
  `GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:295-296`, the
  vector/flux W split at
  `GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:326-346`, the
  first upstream faces at
  `GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:503-533`, and
  the divided concentration RHS at
  `GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:611-623`.

The landed statement is narrower.  The compiled vector program accumulates
HPG, LDF, VOR, KEG and ZAD into `Krhs` at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:141-176`, diagnoses
the depth mean into separate `Ue_rhs`/`Ve_rhs` arrays without overwriting
`Krhs` at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:202-213`, and passes
that unchanged slot into stage 1 at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3.f90:188-202`.

At stage 1 the compiled vector branch builds W for tracers but explicitly
does not use it for momentum at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:326-347`; the
only stage-1 momentum-advection call is under the non-vector guard at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:367-373`.
The vector assignment consumes `Krhs` directly at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:668-674`, before
the ordinary barotropic correction at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:734-759`.

legoesm instead projected the source and then recomputed transport, W/ZAD and
vertical-UP3 contributions after the external solve.  The landing routes the
already complete source directly through the vector assignment and retains
all those recomputations only for the compiled non-vector program.

## Immutable control and one-variable provenance

The incoming tip was run for 2,160 steps with the unchanged certified
from-rest harness, fp64/libm policy, seed 0, six-step snapshots, and tag
`year`.  Its eight registered T rows reproduce the immutable
`phase3/year_equivalence/gyre/` control exactly as stored binary64 values.
The month and 70-row ladder controls were independently regenerated at the
same incoming commit.

The fail-closed producer stamp is
`28e13da8c3b50812b3edec556cb70d36a46b634c`, the preregistration-only child
of incoming scientific tip `a18ba326ec94ab4afbac28de681e69e714621ec8`.
Their `packages/ocean` trees are the identical tree
`7155c8cf56082f21a7edd908f14730d0eb747953`; every one-patch candidate was
branched from that preregistered control.  The registry therefore names the
actual clean harness-producer commit rather than weakening its stamp to the
parent.

Every candidate member manifest names its own clean one-patch commit.  Two
reporting-only descendants require an explicit identity note:

* round 62's scientific production commit is
  `377a3832a034ed0af89cdc3f02b3382c2e8f05ed`; its repaired plant reporter is
  `3f7ec35145b0410285583141e692529300f40a5c`, with the same
  `packages/ocean` tree;
* round 99's stamped candidate is
  `a731e4ef3100c4128bf09e9ff0eaeb7274f9dcb8`; its production tree is the
  same as the original scientific candidate.

The other stamped candidate commits are:

| arm | clean candidate commit |
|---|---|
| `r88_kaa` | `56772e18db135537d6b4f5df03ba5f17ead46935` |
| `r89_assign` | `68214ddd53d15e35db2724a7313abc54243e9852` |
| `r97_rhs` | `5363ee83567e3fcf0475e594c4730ed62048b4e6` |
| `r105_shear` | `5b97aa584e9d46a816b0f748c998438e2cbf022d` |
| `r109_handoff` | `0fa63d7608e26fae41b03d13f31410fcb342c728` |
| `r112_fct` | `fb01c1728733d202d1ca68779ccacbedc93d2afb` |

The r97 and r99 harness commands placed their member directories one level
deeper than the other six arms.  Their preliminary `year_gap.json` files
therefore scored the right snapshots but did not carry a member-admission
block.  Those files are retained.  The authoritative
`year_gap_admitted.json` files were regenerated from the existing snapshots
through the committed Decision-45 admission path; they hash all eight
registered snapshots and bind the clean producer commits above.  No model
step was rerun for this correction.

The first r97/r99 ladder commands also compiled the complete 954-row
diagnostic program.  Those reports and residual sidecars are retained as
additional evidence.  The frozen ranking is deliberately uniform: its
authoritative `ladder_trajectory*.json` artifacts use the same explicit
`--trajectory-only` 70-row lane as the other six candidates and the control.

## Local exactness at the current tip

Every candidate still applies and its registered statement is exact given
NEMO's recorded entry under JIT and eager execution.  These are local
statement proofs, not claims that downstream fields are exact when supplied
legoesm's own upstream state.

| arm | JIT proof | eager proof | non-vacuous production control |
|---|---|---|---|
| `r62_coeff` | production dispatcher seam: matrix 3/3, content 2/2, solve 2/2 BIT | same seven rows BIT | one wet content operand moves by one ULP |
| `r88_kaa` | carried Kaa, W and ZAD 5/5 BIT | source-rounded W walk 14/14 BIT | direct r3 Kaa control moves one cell |
| `r89_assign` | four stage-1/2 assignment rows BIT; production clean count zero | isolated eager U/V BIT | planted production U moves exactly one cell |
| `r97_rhs` | corrected U/V 2/2 BIT through production step | corrected U/V 2/2 BIT through production closure | planted production U moves exactly one cell |
| `r99_wclock` | all 17 kt1 stage-1 outputs plus eight RHS boundaries BIT | W recurrence plus inherited full-RHS U/V BIT | direct r3 control moves exactly one cell |
| `r105_shear` | production shear 3/3 and all captured operands BIT | same rows BIT | step-entry SSH discriminator moves one cell |
| `r109_handoff` | corrected U/V 2/2 BIT through production step | corrected U/V 2/2 BIT through production closure | planted production U moves exactly one cell |
| `r112_fct` | source-literal T/S BIT through production step | source-literal T/S BIT through production eager | consumed first-u plant propagates and fires |

The overbroad full-stage r97/r99 probes hit the documented per-process LLVM
compiler ceiling and are retained as failed attempts.  The admitted r97 proof
uses the focused production handoff closure; the r99 JIT proof uses the
consolidated stage twin with the redundant post-table replay explicitly
omitted, while still scoring every stage output and the separately committed
operator walk.  No failed attempt is presented as a pass.

## Frozen year ranking

Positive `day-240 improvement` means the candidate is closer to NEMO.  The
ordering is exactly the preregistered day-240 ordering; it is not selected
post hoc by day 30 or day 360.

| rank | arm | day 30 T RMS (K) | day 240 T RMS (K) | day 360 T RMS (K) | day-240 improvement (K) | first-over-bar | kt1 AT-BAR losses | landing |
|---:|---|---:|---:|---:|---:|---|---:|---|
| 1 | `r109_handoff` | `6.89043148782590898e-05` | `1.64467402331753935e-02` | `1.12235712478436639e-02` | `+1.69711705452768591e-09` | kt2 U/V -> kt2 U/V | 0 | **PASS** |
| 2 | `r99_wclock` | `6.89048611229354396e-05` | `1.64467419029855622e-02` | `1.12235739180287612e-02` | `+2.73068859191205604e-11` | kt2 U/V -> kt2 U/V | 0 | HELD: day 30 and 360 worse |
| 3 | `r62_coeff` | `6.89048602488395322e-05` | `1.64467419075206220e-02` | `1.12235739634003392e-02` | `+2.27718260914500092e-11` | kt2 U/V -> kt2 U/V | 0 | HELD: day 30 and 360 worse |
| 4 | `r112_fct` | `6.89048687486200015e-05` | `1.64467419160566758e-02` | `1.12235739467723434e-02` | `+1.42357722832109346e-11` | kt2 U/V -> kt2 U/V | 0 | HELD: day 30 and 360 worse |
| 5 | `r97_rhs` | `6.89048740438340842e-05` | `1.64467419165580386e-02` | `1.12235739392417110e-02` | `+1.37344094430780217e-11` | kt2 U/V -> kt2 U/V | 0 | HELD: day 30 and 360 worse |
| 6 | `r105_shear` | `6.89048445214759882e-05` | `1.64467419271117563e-02` | `1.12235739916457765e-02` | `+3.18069182103641879e-12` | kt2 U/V -> kt2 U/V | 0 | HELD: day 360 worse |
| 7 | `r88_kaa` | `3.98171922920888266e-03` | `1.79681504715824003e-02` | `2.88344229640193801e-02` | `-1.52140854128995223e-03` | kt2 U/V -> kt2 T/S/U/V | 0 | HELD: ladder/month/year veto |
| 8 | `r89_assign` | `3.98171936131768572e-03` | `1.79681505729260976e-02` | `2.88344229347229017e-02` | `-1.52140864263364955e-03` | kt2 U/V -> kt2 T/S/U/V | 0 | HELD: ladder/month/year veto |

Because the best single passes, the preregistered top-two pair condition is
false and no post-hoc pair is run.

## Selected candidate: all registered year rows

Decision 45 permits intermediate registered days to worsen but requires
days 240 and 360 not to worsen.  Day 60 is the one worsening year row and is
shown rather than hidden.

| day | before T RMS (K) | after T RMS (K) | after - before (K) |
|---:|---:|---:|---:|
| 30 | `6.89048490148956762e-05` | `6.89043148782590898e-05` | `-5.34136636586328081e-10` |
| 60 | `1.93299736819368750e-04` | `1.93299839204072005e-04` | `+1.02384703254583542e-10` |
| 90 | `1.86450211449135854e-03` | `1.86450172331878482e-03` | `-3.91172573716361272e-10` |
| 120 | `1.05012568195104756e-03` | `1.05012487657951052e-03` | `-8.05371537038537810e-10` |
| 180 | `3.58055101186670896e-03` | `3.58055029328963265e-03` | `-7.18577076314064200e-10` |
| 240 | `1.64467419302924481e-02` | `1.64467402331753935e-02` | `-1.69711705452768591e-09` |
| 300 | `1.35974041773198433e-02` | `1.35974024365470795e-02` | `-1.74077276378359347e-09` |
| 360 | `1.12235739101672668e-02` | `1.12235712478436639e-02` | `-2.66232360289497816e-09` |

## Selected candidate: ladder headline and movement registry

| required row | before max abs | after max abs | before/after class |
|---|---:|---:|---|
| kt2 T | `1.42108547152020037e-14` | `1.42108547152020037e-14` | AT-BAR / AT-BAR |
| kt2 S | `2.13162820728030056e-14` | `2.13162820728030056e-14` | AT-BAR / AT-BAR |
| kt2 U | `2.73771104527739673e-12` | `2.73771104527739673e-12` | DEBT / DEBT |
| kt2 V | `3.28492192214896450e-12` | `3.28492192214896450e-12` | DEBT / DEBT |
| kt3 T | `8.60041971861846832e-07` | `8.65937384020298850e-07` | DEBT / DEBT |
| kt3 S | `6.97944173566611425e-08` | `7.02729110457767092e-08` | DEBT / DEBT |

First-over-bar remains kt2 U/V.  No kt1 AT-BAR row leaves the bar and no row
changes status.  Sixty-two of 70 rows move: 290,796 cells improve, 284,556
cells worsen, and 273,001 cells violate the superseded two-ULP movement bar.
Every moved row is registered below with both oracle residuals and its maximum
model-field movement.

| row | before residual | after residual | class | max field move | improved / worsened cells |
|---|---:|---:|---|---:|---:|
| `GYRE-zco.kt10.after.uu_b` | `8.99484336657971340e-8` | `8.99757315506698185e-8` | DEBT / DEBT | `5.99531821757204320e-11` | 361 / 219 |
| `GYRE-zco.kt10.after.vv_b` | `1.84752761819319267e-7` | `1.84728020489240835e-7` | DEBT / DEBT | `6.56165382445506040e-11` | 252 / 318 |
| `GYRE-zco.kt10.before.S` | `1.29559531814038564e-6` | `1.30835694989173135e-6` | DEBT / DEBT | `3.30412746052388684e-8` | 9271 / 8181 |
| `GYRE-zco.kt10.before.T` | `8.98131854754069536e-6` | `8.97217908146785703e-6` | DEBT / DEBT | `7.82038327429290803e-8` | 9750 / 8211 |
| `GYRE-zco.kt10.before.ssh` | `6.64291944878908636e-7` | `6.67218484562241698e-7` | DEBT / DEBT | `5.66193032909556582e-9` | 279 / 321 |
| `GYRE-zco.kt10.before.u` | `2.86334909990896072e-5` | `2.87547363844913376e-5` | DEBT / DEBT | `4.50434203781366338e-7` | 9503 / 7897 |
| `GYRE-zco.kt10.before.v` | `3.21662513823066029e-5` | `3.23975622864049184e-5` | DEBT / DEBT | `5.93565583569682920e-7` | 8150 / 8950 |
| `GYRE-zco.kt2.after.uu_b` | `5.03983506092544123e-8` | `5.03983506092544123e-8` | DEBT / DEBT | `4.33680868994201774e-19` | 245 / 238 |
| `GYRE-zco.kt2.after.vv_b` | `4.80063497831093380e-8` | `4.80063497831093380e-8` | DEBT / DEBT | `4.87890977618476995e-19` | 242 / 235 |
| `GYRE-zco.kt2.before.S` | `2.13162820728030056e-14` | `2.13162820728030056e-14` | AT-BAR / AT-BAR | `7.10542735760100186e-15` | 1 / 0 |
| `GYRE-zco.kt2.before.T` | `1.42108547152020037e-14` | `1.42108547152020037e-14` | AT-BAR / AT-BAR | `7.10542735760100186e-15` | 0 / 1 |
| `GYRE-zco.kt2.before.u` | `2.73771104527739673e-12` | `2.73771104527739673e-12` | DEBT / DEBT | `1.73472347597680709e-18` | 1269 / 1227 |
| `GYRE-zco.kt2.before.v` | `3.28492192214896450e-12` | `3.28492192214896450e-12` | DEBT / DEBT | `1.73472347597680709e-18` | 1393 / 1433 |
| `GYRE-zco.kt3.after.uu_b` | `1.18048721479339751e-7` | `1.18055625324673709e-7` | DEBT / DEBT | `1.49632753019754428e-11` | 213 / 367 |
| `GYRE-zco.kt3.after.vv_b` | `7.07363997700652045e-8` | `7.07385797638808780e-8` | DEBT / DEBT | `1.77656957612952471e-11` | 270 / 300 |
| `GYRE-zco.kt3.before.S` | `6.97944173566611425e-8` | `7.02729110457767092e-8` | DEBT / DEBT | `2.24459029141144129e-9` | 9315 / 6015 |
| `GYRE-zco.kt3.before.T` | `8.60041971861846832e-7` | `8.65937384020298850e-7` | DEBT / DEBT | `2.76459743986379181e-8` | 10711 / 7036 |
| `GYRE-zco.kt3.before.ssh` | `7.07255898322245100e-7` | `7.07255898322678780e-7` | DEBT / DEBT | `8.23993651088983370e-18` | 289 / 279 |
| `GYRE-zco.kt3.before.u` | `1.22955275766445382e-5` | `1.22913028190035911e-5` | DEBT / DEBT | `3.83444416531854293e-7` | 10141 / 7259 |
| `GYRE-zco.kt3.before.v` | `2.34650172372258270e-5` | `2.34547553134706321e-5` | DEBT / DEBT | `3.82338793993020065e-7` | 8307 / 8793 |
| `GYRE-zco.kt4.after.uu_b` | `1.40789501317222648e-7` | `1.40782212430063877e-7` | DEBT / DEBT | `2.90425291238002869e-11` | 378 / 202 |
| `GYRE-zco.kt4.after.vv_b` | `9.38483140508960742e-8` | `9.38439614126375735e-8` | DEBT / DEBT | `2.80021980267070730e-11` | 241 / 329 |
| `GYRE-zco.kt4.before.S` | `4.72522785344153817e-7` | `4.86354679196665529e-7` | DEBT / DEBT | `1.72062897263458581e-8` | 9799 / 6931 |
| `GYRE-zco.kt4.before.T` | `5.82692681661001188e-6` | `5.82545999705530448e-6` | DEBT / DEBT | `1.25138836182259183e-7` | 10587 / 7298 |
| `GYRE-zco.kt4.before.ssh` | `4.69335130346987955e-7` | `4.69130255937449947e-7` | DEBT / DEBT | `1.97926232109631173e-9` | 303 / 297 |
| `GYRE-zco.kt4.before.u` | `2.05529567587348083e-5` | `2.05477706663008408e-5` | DEBT / DEBT | `3.57044157399399875e-7` | 8154 / 9246 |
| `GYRE-zco.kt4.before.v` | `1.76042212223920247e-5` | `1.76849773129983401e-5` | DEBT / DEBT | `7.41857998658515538e-7` | 10513 / 6587 |
| `GYRE-zco.kt5.after.uu_b` | `1.49277684559443596e-7` | `1.49284908079619708e-7` | DEBT / DEBT | `3.99197081640823015e-11` | 338 / 242 |
| `GYRE-zco.kt5.after.vv_b` | `7.93840125245214023e-8` | `7.93816062780953319e-8` | DEBT / DEBT | `5.42558409088794366e-11` | 277 / 293 |
| `GYRE-zco.kt5.before.S` | `4.28591746981510369e-7` | `4.29654932077028207e-7` | DEBT / DEBT | `6.02162231189140584e-9` | 7397 / 9603 |
| `GYRE-zco.kt5.before.T` | `7.50374877966919485e-6` | `7.47877864526458325e-6` | DEBT / DEBT | `7.29939486632247281e-8` | 7786 / 10097 |
| `GYRE-zco.kt5.before.ssh` | `3.68899188756162399e-7` | `3.70667193931584205e-7` | DEBT / DEBT | `4.40688570357838927e-9` | 330 / 270 |
| `GYRE-zco.kt5.before.u` | `3.70991241753291633e-5` | `3.70512763375263968e-5` | DEBT / DEBT | `7.07809187435636455e-7` | 6908 / 10492 |
| `GYRE-zco.kt5.before.v` | `1.87407282506019124e-5` | `1.85657245908993351e-5` | DEBT / DEBT | `3.24352387193541092e-7` | 9644 / 7456 |
| `GYRE-zco.kt6.after.uu_b` | `1.27792224052791781e-7` | `1.27778894000939950e-7` | DEBT / DEBT | `6.28739202628708638e-11` | 275 / 305 |
| `GYRE-zco.kt6.after.vv_b` | `1.08402871746562587e-7` | `1.08439625359519481e-7` | DEBT / DEBT | `4.82612938328130792e-11` | 311 / 259 |
| `GYRE-zco.kt6.before.S` | `9.27757319857391849e-7` | `9.76131509844435641e-7` | DEBT / DEBT | `4.83741899870437919e-8` | 8178 / 9064 |
| `GYRE-zco.kt6.before.T` | `1.13370560050896074e-5` | `1.13437432567309315e-5` | DEBT / DEBT | `2.29217132385883815e-7` | 8453 / 9484 |
| `GYRE-zco.kt6.before.ssh` | `5.58727563146777656e-7` | `5.58830311346916742e-7` | DEBT / DEBT | `5.77377262016634529e-9` | 316 / 284 |
| `GYRE-zco.kt6.before.u` | `4.16573791435274031e-5` | `4.18240417876815894e-5` | DEBT / DEBT | `3.90316140224528096e-7` | 8559 / 8841 |
| `GYRE-zco.kt6.before.v` | `3.56787824917137106e-5` | `3.56675136264515604e-5` | DEBT / DEBT | `6.50399105900714702e-7` | 7513 / 9587 |
| `GYRE-zco.kt7.after.uu_b` | `1.06959484875816802e-7` | `1.06939411873962359e-7` | DEBT / DEBT | `5.79097503457248752e-11` | 269 / 311 |
| `GYRE-zco.kt7.after.vv_b` | `1.43493745715511656e-7` | `1.43559665441569603e-7` | DEBT / DEBT | `6.82343569667620553e-11` | 263 / 307 |
| `GYRE-zco.kt7.before.S` | `7.53760488692023500e-7` | `7.53390892782590527e-7` | DEBT / DEBT | `5.80583403575474222e-8` | 8375 / 8898 |
| `GYRE-zco.kt7.before.T` | `1.14421208721182666e-5` | `1.14397638988350536e-5` | DEBT / DEBT | `2.10514638609993199e-7` | 8668 / 9300 |
| `GYRE-zco.kt7.before.ssh` | `8.82701506166750499e-7` | `8.83293864292842429e-7` | DEBT / DEBT | `3.75395173540099547e-9` | 253 / 347 |
| `GYRE-zco.kt7.before.u` | `3.05825143003396471e-5` | `3.08566046146041550e-5` | DEBT / DEBT | `6.96825804326513254e-7` | 8875 / 8525 |
| `GYRE-zco.kt7.before.v` | `2.18259029697745292e-5` | `2.17736656534315686e-5` | DEBT / DEBT | `4.87956593284155637e-7` | 8210 / 8890 |
| `GYRE-zco.kt8.after.uu_b` | `1.33678874571067398e-7` | `1.33690015930278475e-7` | DEBT / DEBT | `8.06771914899084841e-11` | 210 / 370 |
| `GYRE-zco.kt8.after.vv_b` | `1.42357009071895219e-7` | `1.42418009295430803e-7` | DEBT / DEBT | `6.10002235355844302e-11` | 316 / 254 |
| `GYRE-zco.kt8.before.S` | `1.02316548833414345e-6` | `1.01999214052739262e-6` | DEBT / DEBT | `1.70424669931890094e-8` | 7948 / 9372 |
| `GYRE-zco.kt8.before.T` | `9.45787083139748574e-6` | `9.44987578321843102e-6` | DEBT / DEBT | `9.99078260122132633e-8` | 8149 / 9810 |
| `GYRE-zco.kt8.before.ssh` | `6.39844431736535935e-7` | `6.39367157946602038e-7` | DEBT / DEBT | `5.95854824422120677e-9` | 304 / 296 |
| `GYRE-zco.kt8.before.u` | `4.15699509193900862e-5` | `4.17538716958182343e-5` | DEBT / DEBT | `4.78636205624577427e-7` | 7678 / 9722 |
| `GYRE-zco.kt8.before.v` | `3.59623020262543266e-5` | `3.60955496261491579e-5` | DEBT / DEBT | `7.11278045791713387e-7` | 8577 / 8523 |
| `GYRE-zco.kt9.after.uu_b` | `9.84088980530289925e-8` | `9.84496771956784589e-8` | DEBT / DEBT | `6.91674746991499667e-11` | 314 / 266 |
| `GYRE-zco.kt9.after.vv_b` | `2.25250001014863296e-7` | `2.25247646180220043e-7` | DEBT / DEBT | `5.34665919936332112e-11` | 256 / 314 |
| `GYRE-zco.kt9.before.S` | `8.60361247134733276e-7` | `8.59850075585200102e-7` | DEBT / DEBT | `6.49608011826785514e-9` | 8863 / 8584 |
| `GYRE-zco.kt9.before.T` | `7.02071718805541423e-6` | `7.00908837814040453e-6` | DEBT / DEBT | `9.43805247288764804e-8` | 9226 / 8740 |
| `GYRE-zco.kt9.before.ssh` | `9.00012351902671447e-7` | `9.00838765753651620e-7` | DEBT / DEBT | `4.30539082597186651e-9` | 300 / 300 |
| `GYRE-zco.kt9.before.u` | `2.92075159027114541e-5` | `2.90107114111915615e-5` | DEBT / DEBT | `6.97504687070804597e-7` | 8913 / 8487 |
| `GYRE-zco.kt9.before.v` | `3.39405389734809618e-5` | `3.42460772755574100e-5` | DEBT / DEBT | `4.58830224587050695e-7` | 8607 / 8493 |

All 30 month rows are retained before/after in the authoritative ranking
artifact.  The complete selected-arm month table is:

| day | before T RMS (K) | after T RMS (K) | after - before (K) |
|---:|---:|---:|---:|
| 1 | `7.56553284967466669e-7` | `7.56696906002323828e-7` | `1.43621034857158859e-10` |
| 2 | `1.05644173125827309e-6` | `1.05798380127843378e-6` | `1.54207002016069836e-9` |
| 3 | `2.18816554715470859e-6` | `2.18804782658862745e-6` | `-1.17720566081136452e-10` |
| 4 | `3.51501147270829225e-6` | `3.51498728069729302e-6` | `-2.41920109992347547e-11` |
| 5 | `4.64960779500600624e-6` | `4.64972914033401809e-6` | `1.21345328011853489e-10` |
| 6 | `5.85319684882198844e-6` | `5.85318875741423628e-6` | `-8.09140775215322350e-12` |
| 7 | `7.20573102515946210e-6` | `7.20567687295096501e-6` | `-5.41522084970843631e-11` |
| 8 | `8.79709852826018711e-6` | `8.79710172855409882e-6` | `3.20029391171188643e-12` |
| 9 | `1.04278344428595213e-5` | `1.04279302805023679e-5` | `9.58376428465966849e-11` |
| 10 | `1.21614093711815178e-5` | `1.21613175920395279e-5` | `-9.17791419899766916e-11` |
| 11 | `1.40578919949392148e-5` | `1.40577689112134556e-5` | `-1.23083725759207357e-10` |
| 12 | `1.65543512962014937e-5` | `1.65562094528600467e-5` | `1.85815665855305445e-9` |
| 13 | `1.81278515740582017e-5` | `1.81274701935239520e-5` | `-3.81380534249763092e-10` |
| 14 | `2.05057256164021024e-5` | `2.05053502687382534e-5` | `-3.75347663849012704e-10` |
| 15 | `2.25760921580208034e-5` | `2.25756051548526744e-5` | `-4.87003168129059244e-10` |
| 16 | `2.71400494110963805e-5` | `2.71395392732119780e-5` | `-5.10137884402416236e-10` |
| 17 | `2.74653583972862872e-5` | `2.74647375712107586e-5` | `-6.20826075528650928e-10` |
| 18 | `3.02676412648950227e-5` | `3.02671454682390374e-5` | `-4.95796655985335450e-10` |
| 19 | `3.25247923407452429e-5` | `3.25241708510242146e-5` | `-6.21489721028289401e-10` |
| 20 | `3.53110483456685139e-5` | `3.53104156700300453e-5` | `-6.32675638468542870e-10` |
| 21 | `3.80929323461515573e-5` | `3.80923628104256037e-5` | `-5.69535725953576252e-10` |
| 22 | `5.14549728694233734e-5` | `5.14535528981006112e-5` | `-1.41997132276213897e-9` |
| 23 | `5.28386738953826127e-4` | `5.28386671284108533e-4` | `-6.76697175941778917e-11` |
| 24 | `4.64740973241067231e-5` | `4.64734000731768310e-5` | `-6.97250929892069115e-10` |
| 25 | `7.10088742239963579e-5` | `7.10077608831813779e-5` | `-1.11334081497993075e-9` |
| 26 | `5.16878876908759873e-5` | `5.16873781265703535e-5` | `-5.09564305633842130e-10` |
| 27 | `5.47203370825482800e-5` | `5.47198466850000066e-5` | `-4.90397548273452302e-10` |
| 28 | `5.72416436757989166e-5` | `5.72411156346226281e-5` | `-5.28041176288526672e-10` |
| 29 | `1.05142608739611755e-4` | `1.05142534024102409e-4` | `-7.47155093464057021e-11` |
| 30 | `6.89048490148956762e-5` | `6.89043148782590898e-5` | `-5.34136636586328081e-10` |

## Cross-card execution and measurement

The executing-card set is derived from resolved recipes, not a hand-written
waiver table.

| resolved card | tracer integrator | momentum advection | outer integrator | executes landed route? | disposition |
|---|---|---|---|---|---|
| GYRE-zco | `rk3_ws` | `vector_invariant` | `forward_euler` | yes | 70-row ladder, month, and year measured |
| generic NEMO-GYRE | `rk3_ws` | `vector_invariant` | `forward_euler` | yes | exact-base three-step certified gate measured |
| LOCK_EXCHANGE-zco | `rk3_ws` | `flux_form` | `forward_euler` | no | non-vector implementation unchanged |
| OVERFLOW-zps | `rk3_ws` | `flux_form` | `forward_euler` | no | non-vector implementation unchanged |
| DINO `nemo_dino_kamm` | `euler` | `vector_invariant` | `forward_euler` | no | WS-RK3 stage program not executed |
| DINO `nemo_dino_kamm_mlf` | `euler` | `vector_invariant` | `leapfrog` | no | WS-RK3 stage program not executed |

The generic NEMO-GYRE before arm is stamped at the exact incoming commit,
not merely a production-identical instrumentation descendant.  All five
certifications remain true.  Thirteen of its 15 state rows move and are
registered here:

| certified row | cells unequal | max absolute movement |
|---|---:|---:|
| `step1_T` | 1 | `1.77635683940025046e-15` |
| `step1_u` | 4361 | `4.33680868994201774e-19` |
| `step1_v` | 3072 | `2.16840434497100887e-19` |
| `step2_S` | 16652 | `5.38696411922501284e-8` |
| `step2_T` | 17667 | `6.63524083677202725e-7` |
| `step2_eta` | 539 | `6.93889390390722838e-18` |
| `step2_u` | 17400 | `1.09748812874532449e-5` |
| `step2_v` | 17100 | `9.41587409806392620e-6` |
| `step3_S` | 17206 | `1.23365360593652440e-7` |
| `step3_T` | 17786 | `9.54833215160988402e-7` |
| `step3_eta` | 600 | `5.01390542621653335e-8` |
| `step3_u` | 17400 | `8.72848408790122265e-6` |
| `step3_v` | 17100 | `8.93452711480096501e-6` |

DINO therefore does not share this statement; its final unit suite is a
preservation check, not a substituted fidelity measurement.  The two tanks
likewise do not execute the vector branch.  No configuration or carried-state
decision is introduced.

ORCA2 remains **UNMEASURED-WITH-SPEC**: instantiate its exact NEMO-identity
card in binary64, resolve whether it reaches WS-RK3 vector stage 1, and, if it
does, compare the full stage-1 source, raw/corrected U/V, transports, W, every
next-stage consumed field, the native short ladder, and a 360-day member
before transferring this GYRE landing.

## Frozen prediction ledger

* **P0 CONFIRMED.** The incoming tip reproduces all eight immutable year rows,
  all 30 month rows, and the certified 70-row ladder at its clean commit.
* **P1 CONFIRMED.** All eight named patches apply alone; no extra production
  hunk appears in any candidate tree.
* **P2 CONFIRMED with the documented bounded-probe qualification.** Every
  statement is BIT under JIT and eager at its registered production seam and
  every plant fires.  The overbroad r97/r99 attempts hit the compiler ceiling
  and are retained rather than relabelled.
* **P3 partly REFUTED.** `r105_shear` is not best.  The predicted day-240
  worsening is confirmed for r88/r89 but refuted for r109/r112.  The frozen
  sub-one-percent response prediction is confirmed for r62/r97/r99.
* **P4 CONFIRMED.** `r109_handoff` is the best single and passes all landing
  criteria; therefore no pair is authorized.
* **P5 CONFIRMED.** The one shared implementation changes only the vector
  stage-1 handoff; configuration, carried state, restart schema, stabilizers,
  NEMO sources, and year harness remain unchanged.

## Mechanical gate and controls

The committed ranking gate consumes exactly eight candidate entries, hashes
every local-proof and cross-card artifact, verifies producer commits, requires
all 70 movement rows and all 30+8 time rows, derives the executing-card set,
and checks the Decision-43/45 predicates.  Its result is:

`STATUS LANDING-CANDIDATE: best=r109_handoff day240_improvers=6`.
The post-landing replay is clean at commit
`14e6e5f8ab1530b8b141bdf6985a4ff1a871c9ac`; its registry SHA-256 is
`2eb097297ce1f5302bb66d2e14c6bdea3183080f657998ca10a7a9b0e3e550da`.
The final shared `packages/ocean` tree and the independently measured
candidate tree both resolve to
`8bfd500f3c0060c5335186b80f389e78f25938a7`.

Removing one candidate, moving the immutable day-240 control, or deleting one
real moved row each prints `STATUS PLANT-FIRED` and exits nonzero.  The
cross-card control also proves that a bare measured-card name cannot discharge
the requirement without its hashed 15-row comparison.

The three gate plants reported, respectively:

```text
STATUS PLANT-FIRED: missing-candidate: candidate registry must be exactly ['r62_coeff', 'r88_kaa', 'r89_assign', 'r97_rhs', 'r99_wclock', 'r105_shear', 'r109_handoff', 'r112_fct']
STATUS PLANT-FIRED: baseline-day240: baseline day 240 moved: 0.01644674193029245
STATUS PLANT-FIRED: missing-moved-row: r62_coeff: moved-row registry is not exact
```

## Independent adversarial review

The required read-only Codex invocation was made with a prompt that tried to
refute registry completeness, one-variable isolation, the Rule-43/45 table,
the shared-card census, and the landing.  The tool could not initialize in
this sandbox; its verbatim verdict was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Thus, **independent review unavailable in-sandbox**.  There is no `DO NOT
SHIP` verdict to override, and the complete failed invocation is retained as
`phase3/round130/codex_review.log`.

## Verification

All completed suites were CPU-only with fp64 enabled:

* The focused changed-path suite covered the Round-130 ranking and stage
  gates, Decision 43, both stage-1 transport suites, both NEMO recipes, DINO,
  and momentum RK3.  Its terminal summary was exactly
  `244 passed, 9 warnings in 559.97s (0:09:19)`.
* After rigidly re-anchoring the 25 citations shifted by this round, the
  citation-gate plus worktree-stamp suite reported exactly
  `26 passed in 3.88s`.
* The required monolithic fidelity+unit invocation collected 8,250 tests and
  reached 96%, but six xdist workers aborted in JAX/XLA compilation in
  unrelated leapfrog, QG-Leith, tracer-integration, and similar tests.  It
  then stopped making progress and was interrupted with exit 130.  It
  produced **no terminal pytest summary**, so it is not represented as a
  pass and cannot supply a complete failure-ID diff.
* A lower-concurrency whole-fidelity retry passed beyond 95% without a red
  test after the citation repair, but a final long test did not return; it
  too was bounded with exit 130 and produced **no terminal pytest summary**.
* Following the repository instruction to split compiler-heavy unit suites,
  the first 60-file batch completed with exactly
  `6 failed, 854 passed, 1 skipped, 2 xfailed, 5 warnings in 594.35s
  (0:09:54)`.  Five failure IDs are in the frozen 87-ID baseline.  The sixth,
  the periodic-channel FCT conservation node, immediately passed alone with
  exactly `1 passed in 4.65s`.  The next batch again suffered an XLA compiler
  abort, this time while compiling an unrelated EKE-source test, and was
  stopped rather than spend the round's CPU budget on ever-smaller shards.

The full-tree limitation is therefore explicit: no reproducible new failure
was found in the completed and focused runs, but this round does not claim a
complete all-tree green summary.  The raw monolithic, fidelity, focused, and
split-unit logs are all retained under `phase3/round130/`.

The receipt citation gate and its shifted-citation plant are reported here:

The unplanted receipt run found 22 citations, zero unmapped citations, zero
failures, an empty map audit, all nine internal self-tests fired, and
`status: PASS`.  Shifting the first endpoint of the registered
`trazdf` matrix range by two lines returned exit 1 with
`status: FAIL` and `SYMBOL-NOT-AT-LINE`; the map audit stayed empty.  The
artifacts are `phase3/round130/citation_gate.json` and
`phase3/round130/citation_gate_shifted_plant.json`.

## Landing and new immutable arms

The final shared `packages/ocean` tree is byte-identical to the independently
measured `r109_handoff` candidate tree.  The new immutable before arms are:

* ladder:
  `phase3/round130/arms/r109_handoff/ladder.json`;
* month:
  `phase3/round130/arms/r109_handoff/month_gap.json` and its member root;
* year:
  `phase3/round130/arms/r109_handoff/lego_seed0_year/` with score
  `phase3/round130/arms/r109_handoff/year_gap.json`.

The original control remains preserved under `phase3/round130/baseline_*`.
No evidence file is deleted or rewritten.

## OPEN — round 131

The landing removes one proven model-only vector handoff, but day-240 T RMS is
still `1.64467402331753935e-2 K`; the campaign is not complete.

1. Start from the new round-130 immutable arms and the landed production tip.
   Do not reuse the old candidate scores as new-baseline scores.
2. Re-prove and year-score the seven remaining held candidates alone against
   the new baseline.  The old ordering makes `r99_wclock` the next positive
   day-240 response, but its day-30/day-360 veto must be remeasured after this
   landing rather than assumed.
3. If no remaining single clears Decision 43+45, resume magnitude ownership
   from the western-jet dynamics.  Require a one-family 360-day sensitivity;
   do not return to isolated last-bit attribution.
4. Keep the exact same-base generic NEMO-GYRE measurement requirement and the
   recipe-derived DINO/tank execution census for any shared statement.
5. ORCA2 remains UNMEASURED-WITH-SPEC as above.  No NEMO acquisition is
   currently needed.
