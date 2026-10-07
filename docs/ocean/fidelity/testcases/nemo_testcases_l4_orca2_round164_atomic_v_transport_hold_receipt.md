# ORCA2 round 164 — atomic V-transport landing hold

Date: 2026-10-07. Base `072fc5eb68`; preregistration `8713c8c87`;
candidate `8ad692f69`; restored tip before this receipt `b716608b9`. Evidence
is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round164/`.
Verdict: **HELD**. The complete source unit still makes the independent rung-0
trajectory non-finite at kt=8, so production is restored exactly to round 163.

All measured ORCA2 numbers are **independent**: hierarchy rung 0 starts from
its own climatological T/S, zero velocity and zero sea surface. The given-entry
rung-7 candidate is unmeasured after the rung-0 terminal refusal. Sea ice, all
six sea-ice selectors and the shipped card's `unmeasured_features` tuple are
unchanged.

## Candidate and terminal result

The candidate promoted the four already-measured statements as one program.
The admitted oracle reads raw face depth at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`, stores the
unmasked completed V transport at `dynspg_ts.f90:568-570`, consumes it in the
separate continuity loop at `dynspg_ts.f90:584-591`, and associates all seven
external-mode fields at `dynspg_ts.f90:761-779`. No public selector or
card/deck field was added.

The production-JIT CPU/fp64/x64/libm rung-0 ladder completed every stage
through kt=7 and exposed kt=8 stages 1-2. Before kt=8 stage 3 could return, the
model refused its own raw-mesh invariant:

```text
raw-mesh e3w_int must contain only finite values > 0
STATUS REFUSE: INTERNAL: CpuCallback error calling callback
```

This reproduces round 155's kt=8 terminal class on the current merged tree.
It refutes R164-P2 before any Decision-96 direction count can be made. The
candidate commit was then reversed; no candidate rung-7, month, GYRE, DINO,
VORTEX, tank or generic-card number is claimed.

The source-boundary rerun was attempted on the restored tree with the admitted
round-144/98 records. Its JAX process disappeared without producing a report
or refusal and left an empty log, so R164-P1 is
**UNMEASURED_THIS_ROUND**. The prior round-155 local bit-exact proof is not
silently promoted to a fresh measurement.

## Restored control

After the retraction, `git diff 072fc5eb68..b716608b9 -- packages/` is empty.
The ordinary rung-0 gate again completes all 200 rows. Its entry identity,
checkpoint count, first debt, cited owner and every row dictionary are
array/value-identical to round 163. The first debt remains kt=1 stage-1 T; the
kt=10 stage-3 S and SSH maxima remain `0.4156673855238111` and
`0.42832517646246693 m` respectively. Thus the failed candidate leaves no
production behavior behind.

## Prediction disposition

| prediction | disposition |
|---|---|
| R164-P1 production equals the complete private arm locally | **UNMEASURED_THIS_ROUND**: the boundary process produced no artifact; only the historical round-155 proof remains. |
| R164-P2 both ten-step ladders complete | **REFUTED**: independent rung 0 refuses between kt=8 stage 2 and stage 3 on non-finite/non-positive `e3w_int`; rung 7 is terminally unmeasured. |
| R164-P3 Decision-96 net improvement | **UNMEASURED_TERMINAL_R164_P2**. |
| R164-P4 halo salinity veto removed | **UNMEASURED_TERMINAL_R164_P2**. |
| R164-P5 month advances beyond step 36 | **UNMEASURED_TERMINAL_R164_P2**. |
| R164-P6 shared cards remain admitted | **NOT_RUN_NO_PRODUCTION_DIFF_AFTER_TERMINAL**. |

## Validation, review and choices

The literal-continuity unit file passes 12/12, including raw-depth mapping,
missing-operand refusal, JIT arithmetic and finite-gradient coverage. The
restored rung-0 control passes and matches all 200 round-163 rows. The failed
candidate log and restored artifacts are hashed in `SHA256SUMS`.

The citation gate and its planted shift are run after this receipt is mapped.
The required separate read-only Codex review is attempted after the receipt
commit; its result is appended without changing the scientific verdict. The
full fidelity battery is not run past a terminal candidate refusal and an
empty final package diff; no downstream landing claim is made.

ASKED choices: the atomic round-163 OPEN item and Decision-96 predicate.
UNASKED choices: empty. No configuration, forcing, carried-state policy,
stabiliser, sea-ice selector or production model statement remains changed.

## OPEN

1. Under the complete private arm, expose the kt=8 stage-2 exit and stage-3
   entry vertical-coordinate operands passively, then name the first cell and
   producer that makes `e3w_int` non-finite or non-positive. Prove the observer
   reproduces the unobserved refusal bit-for-bit.
2. Walk that first finite-to-non-finite boundary one recorded operand at a
   time in compiled stage order. Do not add a stabiliser and do not split the
   four-statement source unit into production choices.
3. Keep the atomic unit and the local halo pair **HELD** until rung 0 completes
   and the Decision-96 row-direction, exact-row and SSH predicates all pass.
