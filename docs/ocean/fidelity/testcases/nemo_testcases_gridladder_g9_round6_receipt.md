# Grid ladder rung G9 — round 6 receipt (gridladder_r51, second resume)

Status: **LANDED ANALYSIS; G9 NOT CERTIFIED.**  The existing round-51 FCT
record is admitted explicitly as **PARTIAL**.  Its five present frames are
finite and its declared sixth frame, `rhs_final`, is reported missing.  The
first NEMO/card departure is the first-midpoint assignment, but the recorded
operands and an isolated transcription of that assignment are exact: the
remaining difference is introduced only when XLA embeds the same arithmetic
in the whole-step graph.  No source association justified a production fix.

- codex review: operator
- independent review: operator (Claude)
- evidence: `gridladder_rounds/round51/`
- admission commit: `a5897cbc67`
- model code changed in this resume: none

## 1. Existing record admitted as PARTIAL

`run_r51_fct_walk.sh --admit-existing` now accepts exactly the payload that
exists, not the payload that was intended.  For each tracer it reads and
checks every value in `entry`, `first_midpoint`, `average_rhs`, `anti_pre`
and `anti_post`: five labels, 19 arrays, all finite, with no unread bytes.
It requires the declared `rhs_final` label and array to be absent and prints
`missing_frame=rhs_final`; a short record at any earlier boundary, a complete
record presented to this PARTIAL route, a non-finite value, a wrong label or
trailing bytes is rejected.

The unchanged passivity loop compares all 42 pre-existing kt-1 and kt-2
binary records with round 46 byte-for-byte before admitting the new files.
The command completed with this final line:

```
G9R51_ACQUISITION_PARTIAL passivity_records=42 missing_frame=rhs_final evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round51/oracle_g9_r51
```

This is no longer `STOPPED_FOR_RECORD`: the owner is before the missing final
frame, so the conditional instruction to repair the observer and define a
new acquisition does not apply.  `ACQUISITION_NEEDED` is therefore `NONE`.

## 2. Boundary walk

The walk uses NEMO's present kt-1 stage-3 arrays and the card's jitted x64
trace from the same exact FCT entry.  NEMO's first divergence is the
left-associated U/V/W flux difference and reciprocal-area multiplication in
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-609`.

| stored boundary | T | S | conclusion |
|---|---:|---:|---|
| first flux divergence replay | 0 unequal | 0 unequal | exact |
| first midpoint | 16 unequal wet values | 13 unequal wet values | first departure |
| averaged-flux RHS | 34 unequal wet values | 25 unequal wet values | downstream |
| pre-limiter faces touching final residue | 21 / 21 cells | 14 / 17 cells | downstream |
| post-limiter faces touching final residue | 21 / 21 cells | 14 / 17 cells | downstream |
| final correction | missing | missing | explicitly unmeasured |

At the final recorded round-46 after-advection boundary the known residue is
still 21 T and 17 S wet cells, with normalized maxima `2.95e-19` and
`1.26e-17`.  Thus the first statement owning their causal chain is the
midpoint assignment itself,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:538`, whose divisor reads the
stored stage-3 `r3t(Kmm)`.

## 3. Why no NEMO-line fix is valid

The source chain was checked at each stored operand:

1. The card's entry SSH and whole-step final SSH are bit-identical to NEMO.
2. NEMO stores the midpoint ratio at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221`; its preceding T-point
   ratio products are the stored `pssh * r1_ht_0` statement at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:257`.
3. Evaluating that source order in an isolated jitted x64 function, with the
   same traced SSH, reference thickness and mask operands, reproduces NEMO's
   `r3t(Kmm)` and all midpoint thickness values exactly: zero unequal.
4. In the whole-step graph only, the resulting Kmm thickness has 16 one-ULP
   differences, all at owned column `(y=4, x=50)`, levels
   `0,1,2,3,5,6,7,8,9,11,13,14,15,17,18,19`; maximum absolute difference is
   `1.4210854715202004e-14`.
5. A G9-only experimental materialization of the stored SSH boundary was
   compiled and compared, then removed: it left the same 16 values and bits
   unchanged.  No alternative reciprocal, divide, sum order or stored-array
   association matches better than the already-exact isolated source program.

The owner is therefore XLA whole-graph arithmetic, not a missing or wrong
source-level association with a cited NEMO statement.  Per the requested
discipline, no model statement is changed.  There is consequently no new
production value test or wiring to plant; the existing exact, jitted,
traced-input, x64 literal-FCT tests remain the applicable controls.

## 4. Rows that remain over the bar

Because this resume changes admission and documentation only, the round-50
both-label kt 1..10 score remains the score of the identical model.  The 26
non-G9 configurations' jaxprs are unchanged by construction: the diff from
the resumed model commit contains no file under `packages/`.

For the independent label, kt 1 has no DEBT; kt 2 remains over the bar in
`ssh`, `u`, `v` and `S`; every one of `ssh`, `uu_b`, `vv_b`, `u`, `v`, `T`
and `S` remains over the bar at kt 3..10.  For the given-entry label, each
single odd step (kt 1,3,5,7,9) has no DEBT, while its paired two-step rows
remain over the bar as follows:

| given-entry pair | rows over the bar |
|---|---|
| kt 2 | ssh, u, v, S |
| kt 4 | ssh, uu_b, vv_b, u, v, T, S |
| kt 6 | ssh, u, v, T, S |
| kt 8 | ssh, u, v, T, S |
| kt 10 | ssh, u, v, S |

G9 is not certified.  Consequently this round certifies none of the ORCA2
lane's proposed fold import: the stage-1 V-mask handling, transport/final
fold exchanges, FCT fold statements and end-of-stage fold/stage pair remain
pending a G9 certification.

## 5. Gates and disposition

The receipt-specific citation gate, run with `JAX_ENABLE_X64=1` and the lane
`PYTHONPATH`, finds 4 citations, 0 failures, 0 unmapped citations and 0
failing map entries.  The default citation-gate suite and private-import
ratchet pass together: **23 passed in 10.11 s**.

ASKED: explicit PARTIAL admission; finite/read-complete present frames;
round-46 passivity; boundary owner; cited-only fix discipline; conditional
score/certification; gates and machine disposition.

UNASKED, offered for revert: none.

UNVERIFIED: the absent `rhs_final` frame itself.  It is not needed to locate
the earlier owner and is never treated as present.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r51/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — PARTIAL record admitted; midpoint is the first unequal boundary, but exact operands and isolated source arithmetic identify the remaining 21 T / 17 S residue as XLA whole-graph arithmetic, so G9 remains uncertified.
