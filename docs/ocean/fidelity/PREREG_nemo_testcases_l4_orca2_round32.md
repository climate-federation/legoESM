# NEMO testcase Lane 4 — ORCA2 card round 32 preregistration

Date: 2026-09-26

Parent: `ee7eb6048`

Status: **PREREGISTERED BEFORE ROUND-32 SCIENTIFIC SCORING.**

Round 32 first lands the one vorticity-denominator statement measured in
round 31: after the F-fold exchange, remaining zero `e3f_0vor` values take the
mesh reference `e3f_3d`.  It then performs the operator's ordered read-outs.
Trajectory results are **independent with Decision-52 SSH**; direct operator
replays are **given NEMO's entry**.  The six sea-ice selectors and the card's
`unmeasured_features` tuple stay frozen.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round32/`.

## Compiled statements

The executing build constructs the masked four-cell reference average at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:914-919`,
exchanges it on the F fold at `:935`, and replaces remaining zero values with
the mesh thickness at `:937`.  The EEN reciprocal consumes that array at
`:734-738`.  The independent `hf_0` read-out is separately tied to its
construction from `e3f_0` at `ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:199`
and the live `r3f` construction at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R32-P0 | The single `:937` substitution repair remains bit-exact and removes the kt=4 refusal. | 0/799,200 construction cells unequal; 40/40 ORCA2 checkpoints; kt=10 stage-3 U/V remain `0.4230544199344075` / `0.6838675521949865` m/s within one ULP of the round-31 arm. | Any construction inequality, refusal, earlier first-non-bit statement, or formerly exact row leaving the bar. |
| R32-P1 | GYRE remains byte-identical at ten steps and through its certified year. | Zero differing trajectory rows and residual arrays; day 30/240/360 remain `6.572574374770603e-05`, `1.644836070117868e-02`, `1.1225660018551306e-02` K and ladder digest `cf06a8fc7d0e90f2`. | Any content, snapshot, pinned number, or digest moves. |
| R32-P2 | The full shared-card and ORCA2 push batteries stay green; the card inventory totals 170 tests. | The exact inherited file list passes 170/170, and the push gate passes its inherited count. | Any new failure or fewer collected card tests without an explained collection change. |
| R32-P3 | Round 24's literal compiled-statement replay remains bit-exact on the landed operator. | 0/411,736 scored U and 0/412,537 scored V cells unequal; its planted violation fires. | Any unplanted unequal cell or a silent plant. |
| R32-P4 | Replacing only the reconstructed F-column depth used by `r3f` with NEMO's carried `hf_0` is secondary and finite. | The first raw EEN output remains bit-identical and the exposed stage-2 change is no larger than round 30's `5.153126997217792e-09` / `2.5500881043307236e-09` m/s2 maxima. | It moves the first raw output, refuses, or exceeds either registered maximum. |
| R32-P5 | Every new or reused gate binds. | Each numerical plant exits non-zero and the citation plant reports `SYMBOL-NOT-AT-LINE`. | Any plant passes. |

Failed predictions remain **REFUTED** and are not rewritten.

## Order and landing rules

1. Commit this preregistration before editing or measuring.
2. Apply only the already-measured `dynvor.f90:937` operand repair.
3. Run the GYRE certified year and ten-step pair before accepting the repair.
4. Run the ORCA2 ten-step ladder, the complete 170-test card inventory, the
   ORCA2 push gate, DINO, lock-exchange and overflow gates.
5. Re-run round 24's compiled-statement replay on the landed operator.
6. Measure the carried-`hf_0` substitution as a separate one-variable read-out;
   it does not land in this round.
7. Run a separate read-only codex review and the citation gate with a real
   plant before the receipt commit.

No stabilizer, clipping, NEMO-source edit, sea-ice edit, score change,
configuration change, carried-state change or acquisition is authorized.

## Choices

ASKED: the B7 addendum explicitly authorizes landing the one repaired
vorticity-denominator statement and orders the six read-outs above.

UNASKED: none.
