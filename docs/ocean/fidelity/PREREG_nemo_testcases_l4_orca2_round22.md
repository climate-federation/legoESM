# NEMO testcase Lane 4 — ORCA2 card round 22 preregistration

Date: 2026-09-25

Parent: `1454e2b3f734c3e2d5a98197b54086e6dc13dbb3`

Status: **PREREGISTERED BEFORE ROUND-22 SCIENTIFIC SCORING.**

Round 22 resolves round 21's first residual by recording exact candidate
arrays, rather than comparing only their scalar row summaries.  It changes no
configuration.  Every number is labelled **independent with Decision-52 SSH**.
The six sea-ice selectors and the card's `unmeasured_features` tuple remain
frozen.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round22/`.

## Executed statements and controlled variable

The admitted NEMO build writes stage 3 immediately after
`stp_RK3_stg(3,...)`, then swaps `Nbb` with that `Naa` level; only the spare
sea-surface level is extrapolated afterward
(`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3.f90:227-244`).
Thus stage 3 and the next step entry must be the same bits for T, S, u, v and
ssh in NEMO.

The merge introduced live NEMO thickness operands at every RK3 lateral-
momentum-diffusion call.  At the step entry and each later stage, production
passes the six Kbb/Kmm thickness arrays into the existing `nemo_e3` operator.
Before commit `94b7761bc7`, those arguments were absent and the operator used
its prior algebraic thickness path.  Round 21's bridge-carried F-thickness
control changes one of the six arrays, but it does not remove this new routing.

Four arms are scored:

1. the archived clean pre-merge ORCA2 tree at `b03f78bb5`;
2. the current tree with round 21's fold-layout and bridge-carried F-thickness
   controls together;
3. arm 2 with only the six live lateral-diffusion thickness arguments withheld,
   reproducing the call boundary immediately before `94b7761bc7`;
4. arm 3 with a one-ULP plant in its exact candidate snapshot.

All arms use the same card, admitted record, forcing, one step, CPU backend,
fp64 policy, and Decision-52 sea-surface entry.  The archived tree is executed
read-only with bytecode disabled; no commit is checked out during the round.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R22-P1 | The exact-snapshot instrument reproduces both admitted summaries. | The pre-merge arm reproduces round 20; the current combined arm reproduces round 21 at every kt=1 row. | Any summary differs: stop for instrument drift. |
| R22-P2 | Scalar summaries hid an earlier exact movement: current combined first differs from pre-merge in kt=1 stage-1 momentum, where the newly routed Kbb lateral-diffusion thicknesses first execute. | Entry arrays are exact; stage-1 u or v is the first unequal exact array; no tracer or ssh field moves earlier. | The first exact difference is later, earlier than stage 1, or in T/S/ssh. |
| R22-P3 | The unregistered third owner is the live lateral-diffusion-thickness routing. | Withholding only those six arguments from the current combined arm restores every exact pre-merge candidate array through the ordinary returned state. | Any exact residual remains; rank its first field and do not re-certify. |
| R22-P4 | The candidate stage-3 trace and ordinary returned state can differ because they are separately compiled, but that observer boundary does not create a NEMO state transition. | Any within-arm difference is reported separately; NEMO's recorded stage 3 and kt=2 entry remain bit-identical in all five fields. | A NEMO field differs across that boundary, which stops the attribution. |
| R22-P5 | The exact comparison and legacy-routing control are non-vacuous. | A one-ULP snapshot plant is refused, and restoring live routing changes at least one exact candidate cell. | Either control is deaf. |

Failed predictions remain **REFUTED** in the receipt and gate.  No production
statement becomes eligible merely because the pre-merge trajectory is restored
by a diagnostic control.  If R22-P3 confirms, the faithful shared statement
stays under Rule 12; the current ladder may be re-certified only after every
moved row is accounted for by the three measured merge contributors.

## Stop and landing rules

- Any failure of R22-P1 stops interpretation of the exact comparison.
- Any NEMO stage-3/kt=2-entry difference stops the round for record or
  instrument reconciliation.
- If the three controls do not restore the pre-merge exact snapshot, name and
  rank the first residual and leave the ladder uncertified.
- Decision 58 and Decision 54 begin only after this reconciliation, in later
  rounds.  Decision 57, the ranked slow-forcing walk, fold-row mask/wind debt,
  and independent year remain untouched.
- No NEMO run is required or permitted in this round.

## Choices

ASKED: instrument the exact kt=1 stage-3 to kt=2-entry transition and walk the
first changed operation before Decisions 58 and 54.

UNASKED: none.  No configuration value, carried-state field, stabilizer,
sea-ice selector, scoring rule, or production statement changes.
