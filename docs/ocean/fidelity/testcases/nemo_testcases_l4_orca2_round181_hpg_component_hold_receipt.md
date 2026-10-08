# ORCA2 round 181 — independent HPG V-component boundary

Date: 2026-10-08. Base `a55bbc186`; measurement commit `f594a391a`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round181/`.
Status: **STOPPED_FOR_RECORD**.

Every scientific number is **independent hierarchy rung 0**. No
given-NEMO-entry result is mixed into this receipt. No production model,
configuration, carried-state rule, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Verdict

The first non-bit component of the raw meridional HPG statement is the
along-surface accumulator `zhpj`. All 1,319 wet levels on the registered 68
northern-fold faces differ: RMS `2.042092669002607e-04 m s-2`, maximum
`1.1053405011865161e-03 m s-2` at `[j=147,i=48,k=28]`. The following local
s-coordinate correction `zvap` is at the fixed floor: magnitude and RMS are
exactly `0.0`; its nine unequal cells are signed zeros. The recorded
`zhpj + zvap` assignment reproduces recorded `sum_v` bit-for-bit. This is the
compiled order and assignment in
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`.

The candidate `sum_v` row reproduces round 179 exactly: 1,319/1,319 unequal,
the same RMS, maximum and argmax. The component boundary is therefore
calibrated to the already-admitted raw-HPG debt rather than a new proxy.

## Record defect and retraction

Prediction R181-P2 is **REFUTED**. Round 180's writer saved `e3w` over only the
owned `DO_2D` domain. At the northern fold, its saved north-halo values are
zero, while the executing compiled HPG expression reads
`e3w_1d(jk)*(1+r3t(ji,jj+1,Kmm))`
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`). Thus the record's
direct component fields are valid and additions-only, but its saved `e3w`
field is not the executed north operand. The literal replay is bit-exact over
both ranks away from the fold and fails only on the northern owned row. The
preregistered north-operand ownership claim remains **UNMEASURED-with-spec**;
the plausible zero is not interpreted as NEMO's value.

This retracts round 180's statement that the record carried every exact HPG
input at the fold. It does not retract its passivity/admission result or the
direct `zhpj`, `zvap`, and `sum_v` fields.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R181-P1 record passive and eligible | **CONFIRMED**: two ranks cover 148x180 exactly once; inherited RHS and all 20 restarts are byte-identical. |
| R181-P2 recorded-input and candidate replay calibrate | **REFUTED** at the fold because the writer omitted the executed north `e3w` expression; off-fold replay is bit-exact. |
| R181-P3 `zhpj` is first | **CONFIRMED** with the numbers above. |
| R181-P4 north-neighbour association owns `zhpj` | **UNMEASURED-with-spec** pending the exact evaluated operands. |
| R181-P5 measurement only | **CONFIRMED**. |

## Replacement acquisition

The committed launcher
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round181_hpg_fold_acquisition/run.sh`
uses the new target `ORCA2_OMIP_L4_R181HPGFOLD`. Inside the executing HPG loop
it records the exact north/current `e3w` and `rhd` operands plus completed
`zhpj`, one self-describing file per rank. The patch removes zero source lines.
Admission parses every field header, requires exact two-rank coverage, and
requires all 20 restarts byte-identical to round 180. The CPU/preprocessor/
Fortran preflight passes and its layout plant exits 69 with
`STATUS PLANT-FIRED`.

## Mechanical checks and review

All seven component-walk plants exit 2 with `STATUS PLANT-FIRED`. The focused
round-181 gate/parser battery passes 15/15. The separate read-only Codex review
was attempted and returned **independent review unavailable in-sandbox**:
`failed to initialize in-process app-server client: Read-only file system`.

No `packages/` file changed, so no ORCA2, GYRE, DINO, tank, trajectory or
month landing gate is eligible in this stopped acquisition round.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round181_hpg_fold_acquisition/run.sh --run`.
2. Admit the exact evaluated operands, replay `zhpj` top-down, and stop at the
   first north/current product or accumulation statement above the floor.
3. Analyse raw HPG, `e3v`, `vmask`, and `r1_hv0` as one cancelling unit before
   any landing. Only after it closes may the held V-transport/halo unit and the
   independent month's step-96 refusal be retested.

The independent 240-step month score remains **UNMEASURED-with-spec**.
ASKED choices are the source-ordered HPG component/operand walk. UNASKED
choices are empty.
