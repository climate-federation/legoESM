# NEMO testcase Lane 4 — ORCA2 card round 31 lateral-diffusion landing and `e3f_0vor` gate

Date: 2026-09-26

Parent: `5d3334700705`

Status: **LANDED (part A) — DYN_LDF NOW HAS ITS OWN FROZEN F THICKNESS.**
**Part B's construction gate reaches its 0-cell bar.**

Rounds 24–30 spent seven rounds splitting one bundle.  This round names the
statement that made the bundle look inseparable: **NEMO's two consumers of
the live F thickness do not share a reference array, and the array those
rounds routed into both of them belongs to only one of them.**  The lateral
diffusion curl multiplies the live stretch onto the MESH reference thickness
`e3f_3d` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`;
the vorticity reciprocal multiplies the same stretch onto `dyn_vor_init`'s
own masked four-cell array `e3f_0vor` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738`.
`e3f_3d` is `e3f_0` by the compiled macro at
`domzgr_substitute.h90:100`, read from the domain file at
`DOM/domzgr.F90:173`.  The carried operand rounds 28–30 substituted is that
mesh array.  Routing it to lateral diffusion is a FIX; routing it to the
vorticity operator REPLACES one NEMO array by a different one, which is why
the same experimental helper cut the tenth step's velocity error by a factor
of sixty in one arm and refused at kt=4 in the other.

Trajectory results are **independent with Decision-52 SSH**; the operator
replay is separately labelled **given NEMO's entry**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple are unchanged.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round31/`.

## 1. Compiled statements

| what | compiled statement |
|---|---|
| lateral diffusion's F curl thickness | `ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123` |
| the vorticity reciprocal that consumes the other array | `ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738` |
| the masked four-cell reference average | `ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:914-919` |
| its F-point north-fold exchange | `ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:935` |
| its zero substitution, by the MESH thickness | `ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:937` |
| the macro that equates the mesh thickness with the domain field | `domzgr_substitute.h90:100` (SOURCE, not the compiled copy) |
| the read of that field | `DOM/domzgr.F90:173` (SOURCE; the preprocessed copy reads it at `domzgr.f90:188` under the active key set) |

The admitted `ocean.output` resolves `ln_dynvor_een = T`, `nn_e3f_typ = 0`
and `ln_dynvor_msk = F`.  The landed legoESM statements are the two
lateral-diffusion call sites in `ocean_model_latlon_cgrid.py:5395-5436` and
`ocean_model_latlon_cgrid.py:5936-5999`, and the reference selection in
`vertical.py:378-476`.

## 2. Which cards execute the changed line

| card | executes `dynldf_lev.f90:123`? | does it move? |
|---|---|---|
| ORCA2-zps | yes | yes — this round's whole measurement |
| GYRE-zco | yes | **no**: 4,590 of 21,120 reference cells differ (max `300.7100172156124` m) and the largest masked viscosity coefficient at those cells is **exactly 0.0**, so the reference never reaches the tendency at any step |
| LOCK_EXCHANGE-zco | no (`lateral_viscosity_e3_weighting` off) | no |
| OVERFLOW-zps | no (`lateral_viscosity_e3_weighting` off) | no |
| DINO `nemo_dino_kamm[_mlf]` | no (same selector off) | no |

GYRE's differing cells are fully dry vertices, where the four-T-cell mask the
coefficient already carries is zero.  That is a stronger statement than a
ten-step comparison: it holds for every day, not only the scored ones.

**The ten-step comparison agrees, to the byte.**  Two GYRE ladders, the
parent from a clean detached worktree at `800602875` and the landed arm at
`4fa07edc7`, differ in exactly three document leaves, all of them the
artifact's own file NAME, plus the worktree stamp.  **Zero content leaves
differ**, and the residual archive's SHA-256 is the same value
`ccd6d39651d8b460...` on both sides.  The 360-day certified year was NOT
re-run this round, so its pinned day 30 / 240 / 360 numbers are quoted
nowhere here; the zero-coefficient argument above is what covers the days the
ten-step ladder does not reach.

## 3. Part A — given NEMO's entry

At NEMO's own recorded kt=2 entry, swapping only the reference moves the
direct lateral-diffusion tendency in **20,647 / 803,640 U** cells (maximum
`7.099144831002128e-06` m s-2) and **20,730 / 804,600 V** cells (maximum
`7.269112645369184e-06` m s-2).  Round 28's whole-bundle arm reported the
same two maxima to four significant figures over far more cells, because it
also swapped the stretch and its mask; this round swaps one operand.

The exposed stage-2 vorticity component keeps round 28's measured parent
digest exactly:
`032cb7d192afb4a60ec5816ab78504ffb96faa5d19d4e83b61c17d1d462247b2`.
A runtime census of the five production builder calls shows the mesh
reference asked for by the two lateral-diffusion call sites only, with the
vorticity path still on its own array.

## 4. Part A — the independent ten-step ladder

Both ladders ran from clean committed worktrees, fp64/libm, CPU, production
JIT, the same deck, record and forcing.

| | parent `800602875` | landed `d0c7fa3ef` |
|---|---:|---:|
| checkpoints | 40 | 40 |
| kt=10 stage-3 U maximum error | `15.365503106245665` m/s | **`0.4230544199344073`** m/s |
| kt=10 stage-3 V maximum error | `42.669598831454074` m/s | **`0.6838675521949863`** m/s |

The parent numbers reproduce round 28's parent to every printed digit, which
is the control that these two ladders are the same comparison round 28 made.

Scored rows: **175 of 200 moved**, **75 toward NEMO** and **100 away** by
maximum absolute error, first at `kt=2:stage1:T`.  **No formerly
bit-identical row left the bar**, and the first non-bit statement is
unchanged at kt=1 stage-1 temperature.  Round 28's consumer-local arm
reported the same 175 rows and the same first moved row.

The 100 rows that move away are reported, not explained away.  They are
scored against a parent trajectory that is itself diverging — its tenth step
carries a 42 m/s velocity error — so a maximum-error row on that baseline is
not a fidelity ranking.  The registered magnitude argument is the one the
order named: the tenth step's velocity error falls by a factor of 36 in U
and 62 in V.  No row labelled **given NEMO's entry** worsens; that control is
section 3 and it is bit-exact.

## 5. Part B — the `e3f_0vor` construction gate

No acquisition carries `e3f_0vor`; the card carries `e3f_0`, `fe3mask` and
`hf_0`.  The gate therefore scores legoESM's production builder against a
literal NumPy transcription of the three compiled statements, on the card's
own carried mesh, one statement at a time.  BLIND SPOT, stated per Rule 2:
the fold permutation itself is legoESM's helper on both sides, so statement 2
scores the ORDER of the exchange, not the permutation, which rounds 21–22
gated separately.

| statement | production, as it was | production, repaired |
|---|---:|---:|
| `:914-919` masked four-cell average | **0 / 799,200** unequal | 0 / 799,200 |
| `:935` after the F-fold exchange | 0 / 799,200 | 0 / 799,200 |
| `:937` after the zero substitution | **335,196 / 799,200** unequal, max `499.9785217898161` m | **0 / 799,200** |

**The bar is met, and by ONE statement, not two.**  The substitution is the
only defect: legoESM filled a fully dry vertex with an UNMASKED four-cell
average of the reference thickness, which on this card is itself zero exactly
where the masked one is, so the fill never fired and 335,196 vertices carried
zero where NEMO carries the mesh thickness.  The ORDER difference the
preregistration also predicted is **REFUTED by measurement**: substituting
before or after the exchange leaves the array identical on this card.

### The 651 m maximum, explained by a named statement

Rounds 29 and 30 reported that NEMO's carried F thickness differs from
legoESM's on 455,904 cells with a maximum of `651.2256783597969` m.  The
carried array they used is `nemo_een_barotropic.e3f_0` — the MESH reference
`e3f_3d`, which `dynldf_lev.f90:123` consumes and `dynvor.f90:734-738` does
not.  Scored against the transcribed `e3f_0vor`, that mesh array differs in
**81,006 / 799,200** cells with maximum **`651.2392608953055`** m at native
index **[97, 40, 29]**, where the mesh thickness is `118.97852178902122` m
and the masked four-cell average is `770.2177826843267` m — a deep
partial-cell column next to a much thinner neighbour, at the bottom level.
The two values are not a defect in either builder: they are two different
NEMO arrays, and rounds 28–30 put one of them where the other belongs.
Their live value is this frozen gap times the live stretch; the two printed
maxima differ by `1.36e-02` m, a relative `2.09e-05`, which is the size of
`r3f` — consistent, and labelled **PLAUSIBLE**, because the same cell was not
independently re-identified in those rounds' arrays.

## 6. Part B — the one EEN arm: THE kt=4 REFUSAL IS GONE

The construction gate met its bar, so the order's one EEN trajectory arm ran:
a separate clean worktree at `5adfc6198` (branch `round31-een-arm`) carrying
part A plus the single repaired substitution statement, on the same record,
forcing, policy and ladder.

| | parent `800602875` | part A `d0c7fa3ef` | part A + B `5adfc6198` |
|---|---:|---:|---:|
| checkpoints | 40 | 40 | **40** |
| kt=10 stage-3 U maximum error | `15.365503106245665` | `0.4230544199344073` | `0.4230544199344075` |
| kt=10 stage-3 V maximum error | `42.669598831454074` | `0.6838675521949863` | `0.6838675521949865` |

**The `raw-mesh e3w_int must contain only finite values > 0` refusal that
held Decision 54 from round 24 to round 30 does not occur.**  The arm reaches
kt=10.  Against part A alone it moves 80 of 200 rows — 1 toward NEMO, 2 away
and 77 at an unchanged maximum — first at `kt=5:stage2:S`, with no formerly
bit-identical row leaving the bar and the first non-bit statement unchanged.
Against the parent it reproduces part A's 175 moved rows, 75 toward and 100
away, exactly.

So the repaired statement is both CORRECT (0 unequal cells against the
compiled transcription, section 5) and SAFE on this card (no refusal, and a
trajectory that differs from part A's only in the last digits of the tenth
step).  **It is NOT landed here.**  The order asked this round to report the
arm, not to land it, and landing it is a second variable that needs its own
GYRE, shared-card and push gates — none of which were run for it.  In
particular the repair changes the vorticity denominator at fully dry
vertices, where GYRE's EEN divisor has its own guard; that is a measurement,
not an argument, and it is round 32's first item with everything else already
in hand.

## 7. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R31-P1 | **CONFIRMED** | rounds 28-30's carried operand is `nemo_een_barotropic.e3f_0`, read straight from the reverted experiment commit; no acquisition carries `e3f_0vor`, and the gate refuses if one appears on the card. |
| R31-P2 | **CONFIRMED** | the exposed stage-2 vorticity digest equals round 28's parent exactly while the lateral-diffusion tendency moves in 20,647 U and 20,730 V cells. |
| R31-P3 | **CONFIRMED** | 40 checkpoints; U `15.365503106245665` to `0.4230544199344073` and V `42.669598831454074` to `0.6838675521949863`, factors of 36 and 62 against a predicted bar of 10. |
| R31-P4 | **NOT MEASURED** | round 24's literal compiled-statement replay was not re-run this round; the given-entry control measured the tendency change and the vorticity inertness instead.  Carried to OPEN rather than claimed. |
| R31-P5 | **CONFIRMED** | GYRE's two references differ on 4,590 cells and the largest masked viscosity coefficient there is exactly `0.0`, so the reference cannot reach the tendency at any step; the confirming ten-step pair differs in zero content leaves with the same residual digest.  The 360-day year was not re-run and its numbers are not claimed. |
| R31-P6 | **HALF CONFIRMED, HALF REFUTED** | the substitution operand is the defect and repairing it alone reaches 0 unequal cells.  The predicted ORDER defect is REFUTED: substituting before or after the exchange leaves the array identical on this card. |
| R31-P7 | **CONFIRMED** | the two arrays differ on 81,006 cells, maximum `651.2392608953055` m at [97, 40, 29], and the repaired builder equals the transcription everywhere.  The arithmetic tying that to rounds 29-30's live `651.2256783597969` m is labelled PLAUSIBLE. |
| R31-P8 | **CONFIRMED** | every plant fires; see section 8. |

The failed prediction is retained, not rewritten.

## 8. Gates, tests and reviews

Plants, each run and each refused:

| gate | plant | result |
|---|---|---|
| `e3f0vor` construction | doubled substitution operand | exit 2, `HELD` |
| `direct-ldf` given entry | corrupted vorticity digest | `HELD`, "the exposed stage-2 EEN component moved" |
| `shared-cards` census | perturbed card reference everywhere | exit 2, `MOVES`, 21,120 cells at coefficient `100000.0` |
| `outcome` | truncated arm | exit 1, refused at the missing kt=10 row |
| `outcome` | inflated kt=10 velocity | exit 2, `HELD`, both maxima |
| citation gate | two-line shift on each of the seven new citations | `SYMBOL-NOT-AT-LINE`, all seven |

The new unit tests were shown non-vacuous by reverting one production call
site in the working tree: `test_each_lateral_diffusion_call_site_asks_for_the_mesh_reference[_step_impl]`
turned red and the rest stayed green; the revert was undone and verified with
`git status --porcelain`.

Batteries, one at a time, each line pytest's own last line:

> ORCA2 push gate (the five files in `autopilot_orca2/autopilot_max.sh`): `127 passed in 389.06s (0:06:29)` — the same count the merge receipt reports.

> Card gates (both DINO gates, the L1 tanks Rule-12 gate, the lock-exchange slow-forcing owner, the overflow barotropic gate, the Rule-12 HPG eligibility gate and the round-34 tank zdf-removal gate): `59 passed in 292.39s (0:04:52)`.  This is a SUBSET of the 170-test card battery earlier rounds quote; the files not in it are named as an OPEN item rather than implied green.

> Rounds 27-31 plus the citation gate: `31 passed in 5.98s`, and `22 passed in 6.06s` for the citation gate with the round-31 unit tests before the review fixes, `9 passed in 3.82s` for the round-31 file after them.

The citation gate passes every citation with zero failures, zero unmapped
citations and zero map entries failing audit.  Thirty-three legoESM
citations were re-anchored RIGIDLY (by six, ten or ninety-five lines) and
three were re-anchored non-rigidly, each for the same stated reason: this
round's own edit widened the span they cite.  No compiled NEMO citation
moved.

### Reviews — BOTH ran, and both found real defects

**codex** (`codex exec --sandbox read-only`): **BLOCK**, two BLOCKERs and two
NITs.  It independently verified the core claim in the compiled sources, the
scope of the change, the byte-for-byte extraction, and four re-anchors.
Findings and disposition:

1. BLOCKER — reverting either call site left all six new tests green.
   **FIXED**: three wiring tests added, and the revert now turns one red.
2. BLOCKER — the runtime census only required a non-empty set of
   consumer-local callers, so a reverted call site could pass.
   **FIXED**: the census is compared against the measured call multiset.
3. NIT — the comparison used numeric equality, so a signed zero could pass as
   bit-exact.  **FIXED**: bit patterns are compared; every number in this
   receipt was re-measured afterwards and none changed.
4. NIT — section 1 called `DOM/domzgr.F90:173` compiled.  **FIXED**: it is
   labelled SOURCE, with the preprocessed line named.

**Claude code-reviewer**: **SHIP WITH CHANGES**.  It confirmed the same core
claim line by line, confirmed the extraction is byte-for-byte, confirmed the
card census rebuilds the SAME coefficient the operator multiplies rather than
a lookalike, and spot-checked six re-anchors.  Two findings:

5. The production call sites were not exercised by any test — the same defect
   codex's first BLOCKER names.  **FIXED** by the same three tests.
6. `substitute_e3f=None` keeps a confirmed defect in production, and the
   receipt attributed that deferral to Decision 54 whose wording does not
   cover it.  **FIXED** in section 9: the deferral is authorized by round
   31's own order, and it is registered as a default that preserves a bug.

Both reviewers independently found the same missing-binding defect, which is
the signal the dual-review rule exists for.  Both reviewed the tree BEFORE
the fixes; the fixes were made in-round and the gates re-run once, as the
order requires.

## 9. Choices

ASKED: Decision 54 authorizes the whole `dyn_ldf` attribution, which is part
A.  It does NOT mention the vorticity operator's zero substitution; that
deferral is authorized by round 31's own order, which says to gate the
`e3f_0vor` construction and to run one EEN arm only if the gate reaches zero.
The review asked for exactly this distinction and it is stated here rather
than attributed to Decision 54.

UNASKED: none.  No configuration value, default, carried state, stabilizer,
score, sea-ice selector, acquisition or NEMO source changed.  The new
reference argument has no default that preserves the old behaviour at a
lateral-diffusion call site: both call sites pass it and the accessor fails
closed when a card does not carry the field.

REGISTERED, not landed: `substitute_e3f=None` keeps the vorticity operator's
measured zero-substitution defect in production this round.  That is a
default that preserves a bug, it is named here, and section 6 reports what
landing it would cost.  It is round 32's first item.

## OPEN

1. **Land the repaired substitution statement.**  Section 6 shows it is
   correct and safe on ORCA2 and that the kt=4 refusal is gone; what it still
   needs is its own GYRE trajectory, shared-card and push gates, because it
   changes the vorticity denominator at fully dry vertices where GYRE's EEN
   divisor has a guard of its own.  The arm is on branch `round31-een-arm`.
2. **Re-run GYRE's 360-day certified year.**  The ten-step pair is
   byte-identical and the zero-coefficient argument covers every step, but
   the pinned day 30 / 240 / 360 numbers were not re-measured this round and
   are not claimed anywhere in this receipt.
3. **Re-run round 24's literal compiled-statement replay** on the landed
   operator, so R31-P4 stops being unmeasured.
4. **Complete the card battery.**  59 tests ran; earlier rounds quote 170.
   Name the missing files and run them.
5. `r3f` still comes from the reconstructed `e3f_0vor` column depth rather
   than NEMO's carried `hf_0` (`domain.F90:149-152` builds that from
   `e3f_0`).  Round 30 measured its effect as a `5e-09` secondary.  One
   statement, not yet substituted.
6. Round 20's ranked slow-forcing producer walk remains open.
7. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
8. The inherited duplicate citation-map literal keys remain open.
