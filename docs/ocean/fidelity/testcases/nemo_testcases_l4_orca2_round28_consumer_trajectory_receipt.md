# NEMO testcase Lane 4 — ORCA2 card round 28 consumer-trajectory receipt

Date: 2026-09-26

Parent: `8237f801c6c928109ca4db049c893f4fb4e28fae`

Status: **HELD — EEN, NOT LATERAL DIFFUSION, OWNS THE KT=4 REFUSAL.**
Routing NEMO's carried raw F thickness only to EEN reproduces the shared
round-26 arm's first movement exactly and reaches the same non-positive
raw-`e3w` refusal while entering kt=4.  Routing it only to lateral diffusion
leaves kt=1 stage-2 unchanged, remains finite through kt=10, and substantially
reduces the late momentum error.  No model statement lands; both experimental
`packages/` edits are reverted and the final `packages/` tree is identical to
the parent.

Every trajectory result below is **independent with Decision-52 SSH**.  The
direct operator result is separately labelled **given NEMO's entry**.  The six
sea-ice selectors and the card's `unmeasured_features` tuple are unchanged.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round28/`.

## Compiled statements and consumer boundary

The admitted build forms lateral diffusion's F curl from live
`e3f_3d*(1+r3f*fe3mask)` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125`.
That result enters the U/V RHS at `:132-140`.

The same build selects EEN vorticity.  It constructs `e3f_0vor` in the active
`nn_e3f_typ` branch and exchanges its north-fold row at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937`.
The EEN operator takes the reciprocal of its live thickness at `:734-738`,
then forms transports and adds the U/V tendency at `:784-802`.  The admitted
`ocean.output` resolves `ln_dynvor_een = T` and `ln_dynvor_msk = F`.

The production implementation called one helper at both boundaries.  Each
round-28 arm changed that helper's source only at one consumer.  EEN-only made
the carried raw operand the helper default while both LDF call sites explicitly
retained the parent reconstruction.  LDF-only kept the parent helper default
and explicitly selected the carried operand at both LDF call sites.  Thus the
arms differ in one consumer, not in configuration, state, forcing, or score.

## Independent trajectory result

The fresh parent and LDF-only artifacts each contain 40 checkpoints
(kt=1..10).  EEN-only contains all 12 checkpoints through kt=3, followed by a
separate kt=4 run that fails closed with
`raw-mesh e3w_int must contain only finite values > 0`.

| arm | checkpoints | moved rows | first moved row | formerly exact rows lost | outcome |
|---|---:|---:|---|---:|---|
| EEN-only raw F | 12 | **45 / 60 observable** | kt=1 stage-2 U | **0** | exact kt=4 refusal |
| LDF-only raw F | 40 | **175 / 200** | kt=2 stage-1 T | **0** | `LADDER_MEASURED` through kt=10 |

The first NEMO mismatch remains kt=1 stage-1 temperature in both arms.  The
EEN-only kt=1 stage-2 maxima are U `0.1007577986233476` and V
`0.11456531655396814` m/s, bit-for-bit the round-26 shared arm's scores.  The
LDF-only values at that checkpoint equal the parent exactly: U
`0.06470386947382581`, V `0.034012848056840184` m/s.

The consumers interact later.  At kt=3 stage 3, EEN-only reaches U
`128.70803778600353` and V `79.77359827617985` m/s, while the shared arm had U
`348.14849703616125` and V `158.39341920935638` m/s.  The exact refusal does
not require that interaction: EEN-only still reproduces it at kt=4.

LDF-only remains finite and changes first one step later.  At kt=10 stage 3 it
reduces the maximum U error from `15.365503106245665` to
`0.42305164302944676` m/s and V from `42.669598831454074` to
`0.6838675738140673` m/s.  These are trajectory sensitivities, not a landing
verdict; Decision 54's full three-statement bundle was not retried this round.

## Consumer-isolation controls

Given NEMO's recorded non-rest kt=2 entry, changing only the F-curl thickness
changes the direct lateral-diffusion tendency in **403,687 / 803,640 U** cells
(maximum `7.099666648286182e-06 m s-2`) and **402,352 / 804,600 V** cells
(maximum `7.269647033321961e-06 m s-2`).  The LDF arm is therefore active, not
a vacuous source edit.

The independently exposed stage-2 EEN component under the LDF-only arm keeps
the parent's digest exactly:
`032cb7d192afb4a60ec5816ab78504ffb96faa5d19d4e83b61c17d1d462247b2`.
Conversely, round 27 already measured the raw-F EEN digest as different while
kt=1 LDF was exactly zero.  Together with the consumer-local source routing,
these controls assign the refusal to EEN without mixing the two operators.

**RETRACTION:** round 26's “F-curl thickness alone owns the kt=4 refusal” is
withdrawn more strongly than in round 27.  The actual F-curl-only trajectory
is finite through kt=10.  The refusal came from the same experimental helper
also changing EEN's live thickness.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R28-P1 | **CONFIRMED** | EEN-only reproduces the shared first U/V scores exactly and the exact kt=4 refusal. |
| R28-P2 | **CONFIRMED** | LDF-only keeps kt=1 stage-2 U/V at the parent and completes all 40 checkpoints. |
| R28-P3 | **CONFIRMED** | Direct LDF changes on the non-rest entry while the LDF-only EEN digest remains the parent digest. |
| R28-P4 | **CONFIRMED** | Neither arm loses a formerly exact row or changes the first NEMO mismatch. |
| R28-P5 | **CONFIRMED** | Both the exact-row plant and the swapped-consumer-label plant are refused. |

No failed prediction was rewritten.

## Gate, review, and tests

The round-28 outcome gate exits 2 with `HELD`.  Its exact-row plant exits 1
with `LDF-only moved a formerly exact row`; its consumer-label plant exits 1
with `EEN arm consumer label is wrong`.  The focused gate battery reports
**21 passed** across the round-26/27/28 and citation-gate tests.

The required separate `codex exec --sandbox read-only` review was attempted on
the committed round diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The citation gate passes all five compiled citations with zero failures, zero
unmapped citations, and zero map-audit failures.  Its rigid +2-line EEN
reciprocal plant exits 1 with `SYMBOL-NOT-AT-LINE`; all nine citation self-tests
fire.

The required `tests/ocean/fidelity -n 12` battery collected 1,871 tests,
emitted the same five inherited failure markers and seven skips as round 27,
then stalled in the inherited final tail after 96%; it was interrupted after a
bounded idle wait without a terminal pytest summary.  Explicit isolated reruns
reproduce the five historical failures: the round-129 stepping-gate stamp, the
stale round-51 live-trace suffix assertion, SI3 scalar-math source provenance,
three unstamped legacy report emitters, and the missing `hires_lane_surface`
case-board row.  No round-28 test fails.

No GYRE/DINO/lock-exchange/overflow landing gate is claimed: this round lands
no model statement, and `git diff 1628376f8 -- packages` is empty at the final
tip.

## Choices

ASKED: Decision 54 and round 27's OPEN item authorize the two consumer-local
raw-F trajectory arms.

UNASKED: none.  No configuration value, default, carried state, stabilizer,
score, sea-ice selector, or NEMO source changed.

## OPEN

1. Decision 54 remains held.  In compiled order, isolate EEN's live-thickness
   denominator from its numerator/transport/tendency, at the EEN statements
   cited above, to find the compensating statement before retrying the whole
   bundle.  Keep the raw-`e3w` refusal unchanged.
2. The consumer-local LDF arm is finite and improves the ten-step trajectory;
   preserve its artifact for the eventual Decision-54 retry, but do not call
   it a landing until the full mask/thickness/metric bundle passes.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on its scheduled independent
   initial-state completion.
6. The inherited duplicate citation-map literal keys remain open.
