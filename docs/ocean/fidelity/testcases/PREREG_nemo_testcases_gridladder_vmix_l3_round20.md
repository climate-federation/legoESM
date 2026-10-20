# GRIDLADDER-VMIX lane — round 20 (vmix_r20): L3 eddy-induced velocity

Status: **PREREGISTERED; ACQUISITION NOT YET RUN**.

Base is the lane-B L2 certificate at `51b2582f5e` plus lane A's certified G9
history through `7a31b67ef9`, joined by merge commit `b3e5011190`.  L3 is one
card, `ORCA2L3-eiv`: G9's tracer-advection card plus L2's complete certified
standard-isoneutral diffusion selection plus ORCA2's resolved eddy-induced
velocity group.  No lane-A arithmetic will be reimplemented.

## Frozen predictions

P1. The L3 deck and reader will resolve exactly five `namtra_eiv` values:
`ln_ldfeiv=T`, `nn_aei_ijk_t=21`, `rn_Ue=0.03`, `rn_Le=200000`, and the
reference-default `ln_eke_equ=F`.  Any missing, extra, mistyped, or changed key
is a refusal.

P2. L3 will differ from G9 only by L2's already-certified isoneutral/MSC path
and the existing GM/Redi Treguier-21 bolus-through-FCT path.  It will differ
from L2 by G9's already-certified tracer-advection selectors and that bolus
path.  The Treguier cap is `0.5 * rn_Ue * rn_Le = 3000 m2/s`; there is no
non-NEMO coefficient floor.  The bolus is tracer transport only and is added
to the stage-3 FCT advecting transport, not momentum or continuity.

P3. Normalized one-step jaxprs for every pre-existing card are unchanged by
the L3 implementation, explicitly including G9, L2, full ORCA2, and DINO.
Only L3 selects the new combination.  Any changed pre-existing-card jaxpr
refutes the round.

P4. The acquisition preflight will establish, without building or running
NEMO, all seven controls from the standing checklist: source patch target and
context; every patched-file compile/syntax proof; exact executable and deck
identity; input manifest; record schema and expected counts; acting witness;
and admission boundary.  Each control has a planted violation that must be
refused.  A preflight that cannot make every plant fail refutes P4.

P5. The acquisition, when the operator runs it, will produce independent
record twins whose source/deck/input identities agree and whose L3 acting
witness differs from both the G9 and L2 admitted baselines in at least one wet
tracer-transport value while leaving the declared non-acting quantities
unchanged.  Exact scores and certification are deliberately unpredicted: this
round stops at `ACQUISITION_NEEDED` and performs no score before an admitted
record exists.

## Frozen falsifiers and boundaries

- L3 construction is refused unless both the G9 tracer-advection selectors
  and the complete L2 isoneutral selector are present.
- Turning off any one of Treguier-21, face averaging, or through-FCT routing
  must change the L3 jaxpr; applying any of them to G9 or L2 must fail the
  selected-only tests.
- A deck with EIV but without standard isoneutral diffusion is refused.
- No NEMO build or run is authorised in this Codex round.  The operator owns
  `--run`; existing outputs are refused unless explicitly admitted by the
  acquisition tool's provenance checks.
- No trajectory score, bar verdict, or certification claim is permitted in
  this round.

## Planned evidence

The committed acquisition entry point will live at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l3_orca2eiv/run.sh`.
Its default action is read-only preflight; `--run` is operator-only.  Evidence
will be written outside git under
`gridladder_vmix_rounds/round20/oracle_l3_r20/`.
