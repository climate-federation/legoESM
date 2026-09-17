# Preregistration: NEMO-testcases L2 GYRE round 103 TKE matrix/RHS statement walk

Date: 2026-09-17. Frozen at the incoming lane tip
`e92f8aff20521b23c6d062a5baf8c46882361efc` (`git rev-parse HEAD` recorded in
the evidence stamp), before any new production-step boundary is exposed and
before any physics or configuration change. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round103/`.

## What changed since the round-102 receipt was written

Three facts post-date that receipt and are carried into this round as
premises, not as claims to be re-established:

1. The GYRE NEMO-identity card now selects
   `tke_langmuir_evaluation="nemo_literal"` (user decision 42, card only,
   library default unmoved) together with the safe-sqrt gradient guard. The
   production arm of the TKE walk is therefore the literal arm.
2. Round 102's attribution of its 3,223-cell post-Langmuir difference to the
   ordered `zpelc` recurrence is **REFUTED by measurement** (operator note Q).
   The measured owners are the `imlc` mixing-layer index and the missing
   surface `tmask`/`wmask` gating, which are mutually redundant; the ordered
   recurrence is inert. This round does not walk that recurrence.
3. The Langmuir switch buys ATTRIBUTION, not accuracy at the block exit: the
   post-sweep row is 979 cells on both arms and the literal arm is marginally
   worse in the last bits.

## Calibration already run before this preregistration

Per the campaign's instrument rule, the existing consolidated gate
(`nemo_testcase_l2_gyre_round46_kt2_stage_gate.py --mode stage-tke-walk`) was
re-run at this tip before anything new was built, purely to reproduce known
values. Artifact `instrument_repro.json` / `.log`. It reproduces round 102's
literal-arm rows exactly on the `NEMO_TKE_RECORDED` (given-NEMO-entry) arm:

| boundary | round 102 literal arm | this tip, production arm |
|---|---|---|
| `en_entry` | 0 / 0 | 0 / 0 |
| `en_after_boundaries` | 0 / 0 | 0 / 0 |
| `en_after_langmuir` | 0 / 0 | 0 / 0 |
| `rhs_pre_sweep` | 11,993 / `5.488912518947231e-10` | 11,993 / `5.488912518947231e-10` |
| `en_post_sweep` | 979 / `6.809688229969524e-12` | 979 / `6.809688229969524e-12` |

Cells are `unequal / max abs`. No new science was read before this file was
committed.

## The open question

The first non-bit boundary given NEMO's recorded stage entry is
`rhs_pre_sweep`, which the current instrument reports as one 75-line block,
`zdftke.f90:399-473`. This round subdivides that block into its individually
recorded compiled outputs and names the FIRST statement whose output is not
bit-identical, through the real production step (`_step_jitted`), driven from
NEMO's recorded stage entry.

No new NEMO acquisition is needed. The independently admitted Round-59 record
`round59/oracle_tke_operands/oracle_tke_operands_kt00000002.bin` already
stores `matrix_upper`, `matrix_lower`, `matrix_diag`, `rhs_pre_sweep` and
`pdlr`, written at `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:172-190`
from the call `GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:472`,
which is the statement immediately preceding the Round-101 RHS boundary
callback at `:473`. The two records therefore describe the same instant.

## Registered exclusion: `p_pdlr`

`p_pdlr` is written at `zdftke.f90:421` and is, in strict compiled execution
order, the first output-bearing assignment after the Langmuir callback at
`:395`. It is registered here as **UNMEASURED-WITH-SPEC and excluded from the
`en` walk**, with the reason cited rather than assumed: `p_pdlr` is read at
exactly one place, `zdftke.f90:712` inside `tke_avn`, and never enters `en`.
It is a separate consumer chain (the `avt` Prandtl multiplier) and is named in
this round's OPEN section for its own walk. Exposing it through the production
step requires threading a return out of the model's Prandtl helper, which is a
larger diff than this round's question needs.

## Frozen predictions

**P1 — source inventory.** Between the Langmuir callback (`zdftke.f90:395`)
and the RHS callback (`:473`), exactly these output-bearing assignments
execute on this card, in this order:
`p_pdlr` (`:421`, excluded above), `zd_up` (`:434`, value `:429-430`),
`zd_lw` (`:435`, value `:431-432`), `zdiag` (`:436`), and the `en` right-hand
side (`:439-442`). The wave-coupled surface block at `:451-468` does NOT
execute, on two independent conditions: `cpl_phioc` is set `.TRUE.` only in
`sbccpl.f90:629` (coupled runs, which GYRE is not) and `ln_phioc = .false.`
in `EXP00/namelist_ref:593`.
REFUTED by any other executing assignment writing a recorded array between
those two callbacks, or by `cpl_phioc .AND. ln_phioc` resolving true.

**P2 — first non-bit statement.** Given NEMO's recorded stage entry, through
the production step, scored over every cell NEMO's solve consumes
(`jk = 2:jpkm1`), the first non-bit output in compiled order is the `en`
right-hand side assignment at `zdftke.f90:439-442`; `zd_up` (`:434`),
`zd_lw` (`:435`) and `zdiag` (`:436`) are each BIT with exactly 0 unequal
cells.
REFUTED if any of `zd_up`, `zd_lw`, `zdiag` has one or more unequal cells, in
which case that earlier statement is the named owner instead.

**P3 — ownership.** The `en` RHS inequality is INHERITED from the model's own
shear production `p_sh2`, not owned by the assignment's arithmetic. The
model's production `p_sh2` is non-bit against the Round-59 recorded `sh2`, and
every cell where the production RHS is unequal is a cell where `p_sh2` is
unequal (the RHS-unequal set is a subset of the `p_sh2`-unequal set).
REFUTED if the production `p_sh2` is BIT (then the owner is inside the
assignment or in another operand), or if any RHS-unequal cell has a bit-equal
`p_sh2`.

**P4 — landing.** No candidate fix lands this round. If P2 and P3 both hold,
the owner is the shear production, which is computed OUTSIDE the TKE block by
an upstream operator; under decision 41 nothing downstream lands while an
upstream stage has an owned non-exact output, so the round's status is HELD.
REFUTED if P2 is refuted in a way that names an in-block owned statement with
a cited one-line fix, in which case the Rule-12 ladder (kt=1..10, 954 rows)
and the days 1-30 arm are run before any landing, against the immutable
round-96/97 before arm (kt2 U/V `2.7377110452773967e-12` /
`3.284922138989399e-12`; kt3 T/S `1.627497246303733e-4` /
`6.327735185607253e-6`; day-30 T RMS `1.2397011295506804e-2 K`).

## Controls that must fire

The new sub-boundary rows are worthless unless they can fail. A plant
`stage-tke-matrix-ulp` will advance exactly one bitwise cell of the recorded
`matrix_diag` reference on the binding `NEMO_TKE_RECORDED` arm; the clean row
must have zero unequal cells, the planted row exactly one, and the gate must
exit nonzero. The plant must name a non-null target (the round-102 null-target
failure is not repeated), and it must fire through the production step, not
against an isolated closure (note L-amend).

## Scope limits

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified and neither
`makenemo` nor `mpirun` is run. No configuration field, default, scheme
selection, threshold or carried state changes. The year harness, the
reconciliation gate, the freshwater pair and the #1484 guard are read only.
Held patches under `manifests/` are not re-evaluated: none of them names a
statement in `zdftke.f90:424-443`.

---

## ADDENDUM 1, 2026-09-17, after an independent review and before the receipt

Two corrections to the text above. The predictions themselves are NOT touched;
they stand exactly as frozen and are reported against as written.

**A1.1 — the record provenance in "The open question" is WRONG.** That section
says the Round-59 arrays were written "from the call
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:472`". They were not.
The Round-59 record was produced by the Round-59 binary, whose write call is
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:463`; line 472 belongs to
a different binary. The conclusion the section draws survives, but for a
reason that had to be measured rather than assumed: the two builds' compiled
`zdftke.f90` differ in ZERO lines once the `r54_`/`r101_` recorder call lines
are removed, so the two records do describe the same statements at the same
instant. The Round-59 build carries those statements at `:417`, `:420-421`,
`:422-423`, `:425-427` and `:430-433`.

**A1.2 — P3's test is weak, and a stronger one is added POST HOC.** P3 as
frozen is a SET-INCLUSION test: it asks whether every cell where the
production right-hand side differs is a cell where `p_sh2` differs. An
independent reviewer pointed out, correctly, that this cannot be causal,
because the same statement also consumes `p_avt`, `rn2`, `dissl`, the
post-Langmuir `en` and `wmask`, and coincident errors in those would pass it.
P3 will still be reported exactly as frozen. In addition, a one-variable swap
is added as post-hoc evidence: rebuild the statement from NEMO's own recorded
operands, substitute ONLY the production `p_sh2`, and report what that alone
reproduces. It is registered here as POST HOC and will never be presented as
something this round predicted.
