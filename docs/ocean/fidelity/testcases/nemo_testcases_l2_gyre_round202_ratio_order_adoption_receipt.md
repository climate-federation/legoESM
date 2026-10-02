# GYRE round 202 — adopting the shared stage-one tracer ratio order

**Date:** 2026-10-02
**Lane:** `fidelity/nemo-testcases-l2-gyre-codex2`
**Base:** `52e9a3112ee5` (round 201)
**Adopted from:** ORCA2 lane commit `a0b2f7a5da06f416e530361355df8603b7826a3d`
(2026-09-27, "preserve shared stage-one ratio order"), admitted against the
GYRE gate by ORCA2 round 102
(`nemo_testcases_l4_orca2_round102_gyre_admission_receipt.md`).
**Disposition:** **HELD — the statement is transcribed and measured, but the
ORCA2-registered year numbers DO NOT reproduce on this lane past day 90, so
nothing was re-pinned and the model change does not land.**

## The statement, and its citation

NEMO's RK3 stage 1 needs the thickness ratio `1 + r3t` at the stage level
`Kaa = N+1/3`.  It does **not** interpolate the free surface and then form one
ratio from it.  It first builds the *after-level* ratio arrays from `ssha`
(`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:167`, the call
`dom_qco_r3c_RK3`), and only then interpolates the two endpoint ratios
(`:177`):

```
r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)   ! at N+1/3 (Kaa)
```

The whole block is `GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`;
GYRE resolves the `np_LIN`/`np_HYB` branch, which is the one quoted.  The two
orders are real-equivalent and not bit-equivalent: SSH-first rounds one
product of an interpolated operand, ratio-first rounds two products and their
sum, exactly as the Fortran does.  This is the same compiled statement the
ORCA2 lane cited from its own build at the same line range.

legoESM now transcribes that association in
`nemo_r3t_rk3_stage1_stretch` (`packages/ocean/legoesm/ocean/eos.py`), and the
stage-one tracer weight `_qt_13` in
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` calls it.
The other three tracer weights (`_qt_b`, `_qt_12`, `_qt_aa`) are untouched.

## What was taken, and what was deliberately not

| file in `a0b2f7a5d` | taken | why |
|---|---|---|
| `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` (9 lines) | **yes** | the call site of the shared statement |
| `packages/ocean/legoesm/ocean/eos.py` (34 lines) | **yes** | the shared transcription itself |
| `tests/ocean/unit/test_nemo_ws_tracer_rk3.py` (40 lines) | **yes** | its unit control |
| `docs/.../nemo_testcases_l2_gyre_phase3_round8_receipt.md` | no | the ORCA2 lane's edit of its own receipt text; this lane's copy is current |
| `scripts/.../orca2_l4/nemo_testcase_l4_orca2_round45_qco_rk_gate.py` | no | an ORCA2-lane gate; the file does not exist on this lane |
| `scripts/.../nemo_testcase_l2_gyre_decision43_gate.py` | no | the ORCA2 lane's version of the Decision-43 gate, with ORCA2 census rows |
| `tests/ocean/fidelity/test_nemo_testcase_l2_gyre_decision43_gate.py` | no | its test, same reason |
| `scripts/.../nemo_testcase_receipt_citation_gate.py` | no | the ORCA2 lane's citation-map edit; this lane adds only the entry its own receipt needs |

Each refused file is an ORCA2-lane artefact and lives on that branch.  Nothing
else in the tree moved: no default, no card, no deck, no threshold, no carried
state, no scheme selector.

## The certified GYRE ladder — unchanged

The full certified gate (`nemo_testcase_l2_gyre_phase3_gate.py --max-step 10`)
was re-run with the statement applied and compared row by row against round
201's certified report with the shared oracle-relative gate
(`nemo_testcase_offline_compare.py`):

```
OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=954 max_worsening_ulps=0
  first_over_bar={'T','S','u','v','ssh'} kt=3 -> unchanged
```

**954 rows, 0 moved, 0 ULP, no row status change, no violation.**  The ten-step
GYRE trajectory does not see this statement at all; the first ten steps are
below its rounding difference.  That is why the year, not the ladder, is the
binding measurement here.

## The from-rest year — WHY THIS ROUND IS HELD

The order of operations this round was: measure the year, check it against the
numbers ORCA2 round 102 registered, and only then re-pin.  It failed at the
check, so no pin moved.

A fresh seed-0 member ran the full 2,160 steps and wrote all 360 daily
snapshots (`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360
--snap-steps 6 --tag r202`, 1,474.6 s, member
`phase3/year_fromrest/lego_seed0_r202`).  The day-by-day gap against NEMO was
then scored with the committed scorer (`nemo_testcase_l2_gyre_year_owners.py
--day-gap`), on BOTH arms, same scorer, same NEMO restarts, same eight days.

| day | this lane, BEFORE | ORCA2 certified | agree | this lane, AFTER | ORCA2 merged | agree |
|---:|---:|---:|:--:|---:|---:|:--:|
| 30 | `2.3432510206121264e-06` | `2.3432510206121264e-06` | yes | `2.3432465132112266e-06` | `2.3432465132112266e-06` | **yes** |
| 60 | `1.4793247420315405e-05` | `1.4793247420315405e-05` | yes | `1.4793247973304582e-05` | `1.4793247973304582e-05` | **yes** |
| 90 | `1.6332637650962138e-05` | `1.6332637650962138e-05` | yes | `1.633271203963844e-05` | `1.633271203963844e-05` | **yes** |
| 120 | `1.0965898339728307e-04` | `1.0965898339728307e-04` | yes | `1.0965906837581848e-04` | `1.0965907352116351e-04` | **NO** |
| 180 | `6.115333823687429e-05` | `6.115333823687429e-05` | yes | `6.115335288161464e-05` | `6.115335539300055e-05` | **NO** |
| 240 | `6.581707093530567e-05` | `6.581707093530567e-05` | yes | `6.58170624837412e-05` | `6.58170609494473e-05` | **NO** |
| 300 | `5.466049869672802e-05` | `5.466049869672802e-05` | yes | `5.466050113289237e-05` | `5.466049845187051e-05` | **NO** |
| 360 | `5.407735418221895e-05` | `5.407735418221895e-05` | yes | `5.407736527246344e-05` | `5.4077419367442036e-05` | **NO** |

Read it as a diff, which is the whole point of measuring both arms with one
scorer:

* **Before the change the two lanes agree on all eight days to every printed
  digit.**  The pre-adoption GYRE year is the same number on both trees.
* **After the change they agree on days 30, 60 and 90 and part company from
  day 120.**  Day 30 even agrees at the file level: this round's `day030.npz`
  has SHA-256 `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`,
  which is exactly the digest ORCA2 round 102 registered for its merged
  member.  Day 240 (`a63befc30bf03b443e54a51a0dc0541a186a0d20106e3ba94133b1489f488534`)
  and day 360 (`dcb7bc46c8bc75bd215b4752c8145b9025d00da38a6fdac8c6babd3cd6074899`)
  do not match ORCA2's registered digests
  (`8b9cd60475626373d9a2fec0baa508f06c1f91aed4c3d7a24fc5d8b3f5878c8a`,
  `e3e0a068346c7866f0a318bb32141f2fb326cc05c158a1c95bc39b091f585e25`).
  The digest check and the rms check fail together on the same days, so this
  is not a scoring artefact.

The round order said, in one line, that any differing digit stops the round.
It differs, so the round stops: **no certified GYRE number and no certified
digest was re-pinned, and the production code does not carry the statement.**
The three rows the order asked to register are therefore NOT registered.

**What is and is not established.**  CONFIRMED: the transcription itself
reproduces ORCA2's first three registered days exactly, day 30 bit for bit —
the statement is the same statement on both lanes.  NOT ESTABLISHED, and NOT
investigated this round because the order forbids it: why the two trees part
company from day 120 when their pre-adoption years are identical.  The
obvious candidate is PLAUSIBLE only — the ORCA2 number comes from a MERGED
tree (ORCA2 round 100's merge) that carries lane content this tip does not,
and a difference that is inert on the old trajectory need not be inert on a
new one.  Run-to-run nondeterminism is the competing explanation and is not
excluded here, though rounds 199 and 201 both reproduced 360 of 360 snapshots
byte-identically on this host, which argues against it.  The discriminating
measurement is a second independent 360-day member on this tip with the
statement applied: identical numbers refute nondeterminism and leave the tree
difference; different numbers settle it the other way.  It costs about
25 minutes and is the first item of the next round.

For the record, the old pin does fail against the new year, which is the
non-vacuity the re-pin step would have needed: 344 of 360 daily snapshot
files differ from the certified carried arm
(`phase3/round202/gyre_year_byte_identity.txt`).  That number is reported as
a measurement, not as grounds for a pin.

## The other cards — all inert

| card | reference | result |
|---|---|---|
| `VORTEX-zco` (flux) | round 201 | **PASS**, 50 rows, **0 moved**, `max_worsening_ulps 0`, `first_over_bar` unchanged `{T,u,v,ssh} kt=2` |
| `VORTEX_VEC-zco` (vector) | round 201 | **PASS**, 50 rows, **0 moved**, `max_worsening_ulps 0`, `first_over_bar` unchanged `{u,v,ssh} kt=2` |
| `LOCK_EXCHANGE-zco` tank | round 199 | **PASS**, 50 rows, 0 moved, `first_over_bar` unchanged `{u} kt=4` |
| `OVERFLOW-zps` tank | round 199 | **PASS**, 50 rows, 0 moved, `first_over_bar` unchanged `{T,u} kt=2` |

Both VORTEX registries are 0 of 50, which is what the round order expected,
and the ratchet is green on both.

## Gates

| gate | result |
|---|---|
| GYRE certified kt=1..10 ladder vs round 201 | **PASS**, 954 rows, 0 moved, 0 ULP |
| GYRE 360-day from-rest year vs the registered ORCA2 column | **FAIL on days 120/180/240/300/360** — the reason this round is HELD |
| `VORTEX-zco` / `VORTEX_VEC-zco` certified 50-row registries | **PASS**, 0 of 50 each |
| `LOCK_EXCHANGE-zco` / `OVERFLOW-zps` tanks | **PASS**, 0 of 50 each |
| cellwise two-ULP ratchet plants (flux card) | `worsen-3ulp` exit 1 (`max_worsening_ulps=3`), `at-bar-to-debt` exit 1 — both red on the same inert pair the unplanted run passes |
| receipt citation gate, `DEFAULT_RECEIPT` | **PASS** exit 0, 274 citations, `unmapped_citations []`, 0 failures, all nine self-test plants fired |
| generic NEMO-GYRE recipe gate, push battery | run by `land.sh` |
| DINO from-rest month gate | run by `land.sh` (reference `2.053801168e-03` K, bar `2.244317642e-03`) |

Evidence, all under `phase3/round202/`: `gyre_ladder_after.{json,log}`,
`gyre_ladder_compare.json`, `gyre_year_r202.log`,
`gyre_year_byte_identity.txt`, `gyre_day_gap{,_before}.{json,log}`,
`traj_{VORTEX-zco,VORTEX_VEC-zco,LOCK_EXCHANGE-zco,OVERFLOW-zps}_after.{json,log}`,
`ulp_*.json`, `ulp_plant_*.{json,log}`, `ratchet_plants.txt`,
`citations_default_receipt.{json,log}`, and the trajectory-only ladder arm
`trajonly_*` kept because it is what first showed the gate's 70-row subset.

## Non-vacuity

* **The ULP ratchet plants are red on the very pair the unplanted run
  passes**: `worsen-3ulp` reports `max_worsening_ulps=3` and
  `at-bar-to-debt` reports a crossed row, both exit 1
  (`phase3/round202/ratchet_plants.txt`).
* **The citation gate's nine internal plants all fired** (generic anchor,
  widened extent, reversed range, endpoint shifted alone, and the rest), and
  the gate found 0 failures only after the re-anchor — before it, it reported
  7 receipt failures and 27 map entries off by exactly the five lines the
  model hunk adds.  That is the gate doing work, quoted rather than claimed.
* **The unit control refuses the old order.**  The adopted test builds the
  SSH-interpolated-first value explicitly and asserts the new helper does not
  equal it; the independent reviewer re-derived both orderings outside the
  harness and measured a real one-ULP difference, so the assertion cannot pass
  vacuously.
* **The year comparison is two arms through one scorer.**  The before column
  was re-scored this round from the certified carried arm with the same
  committed scorer, the same NEMO restarts and the same eight days as the
  after column, so the cross-lane agreement before and disagreement after is
  a diff and not a protocol difference.
* **The old pin fails against the new year** — 344 of 360 snapshots differ —
  so a re-pin would not have been vacuous had the numbers justified one.

## Independent review

One fresh `code-reviewer` subagent, no context from this round, on the diff
only.  Verdict **APPROVE**, with two non-blocking notes, both recorded here
because they are real:

1. The sibling helper floors its stretch at `1e-6` as a legoESM-only safety
   net and the new one does not.  Inert on GYRE's closed box (no wetting and
   drying) and arguably more faithful to the Fortran, but it is a latent
   inconsistency between two helpers feeding the same tracer-weight tuple.
2. The companion `r3u`/`r3v` stage-one ratios are NOT transcribed and still
   interpolate the free surface first.  The adjacent code comment says the
   vector-invariant GYRE arm does not consume those factors, so this is
   plausibly inert on this lane, but the commit message did not say so.

It re-derived the association against the compiled source (including that
NEMO's `r1_3`/`r2_3` are divisions, `stprk3_stg.f90:53-54`), confirmed the
single production caller and the three untouched weights, and ran the unit
battery itself: **32 passed in 1754.39s**, exit 0.

Both notes are carried into OPEN rather than acted on, because this round
lands no production change.

## Choices made this round

| choice | ASKED / UNASKED |
|---|---|
| Take only the two model hunks and the unit test, refuse the five ORCA2-lane artefacts | the round order named exactly this |
| HOLD rather than land, because the year numbers differ | the round order's own stop condition |
| Revert the citation re-anchor along with the model hunks | mechanical; the re-anchor exists only because the hunks shift those files |
| Add the production GYRE build to the citation map | required so this receipt's own citation is audited rather than unmapped; additive, no existing entry or check changes |
| Compare the ladder against round 201's FULL certified report rather than the 70-row trajectory-only subset | like-for-like; the trajectory-only arm is kept as evidence |
| Re-score the before arm this round instead of quoting round 191's three days | controlled comparison; it is what makes the cross-lane diff readable |

**UNASKED list: empty.**  No default, scheme, bound, threshold, deck value,
card line or carried state was changed, and production code is byte-identical
to the round-201 tip.

## OPEN, in order

1. **Settle the year divergence with one measurement, not an argument.**  Run a
   second independent 360-day member on this tip with the held patch applied.
   Identical numbers refute run-to-run nondeterminism and leave the tree
   difference between this lane and ORCA2's merged tree; different numbers
   settle it the other way.  About 25 minutes, no NEMO run.
2. **If it is the tree difference**, diff this tip against the tree ORCA2
   round 102 measured on and name the statement that is inert on the old GYRE
   trajectory and not on the new one.  Rounds 198 and 199 both landed shared-path
   statements and both proved GYRE byte-identical BEFORE this change, which is
   the obvious place to look and not yet a claim.
3. **Then re-run this round's step 3** — register the moved year rows and
   re-pin the certified GYRE numbers and digests — against whichever column
   survives.
4. The reviewer's two notes: the missing `1e-6` floor in the new helper, and
   the untranscribed `r3u`/`r3v` stage-one ratios
   (`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:160-179` covers all three).

## UNVERIFIED

* Why the two lanes' years part company from day 120 is **not measured**; both
  candidate explanations are named above and neither is confirmed.
* The day-gap scorer was run for eight days only, the days NEMO has restarts
  for; no claim is made about any other day.
* The focused unit battery was run by the independent reviewer on the applied
  patch, not on this landing tree, which carries no model change.
