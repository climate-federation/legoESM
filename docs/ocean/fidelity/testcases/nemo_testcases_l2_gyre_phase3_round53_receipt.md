# NEMO testcase L2 GYRE phase-3 round-53 receipt

Date: 2026-09-11. Base: `8ecdfd459234`. All integrations were CPU,
production-JIT fp64 and used independent, pre-existing NEMO records. NEMO was
neither run nor modified. The committed preregistration predicted that the 57
Round-44 worsened rows would disappear or be confined to explainable kt>=3
rows; any row whose worsening exceeded the kt=2 improvement was the falsifier.

## Compiled-source decision

GYRE executes the vector arm in source order HPG, LDF, VOR, WZV, KEG, ZAD
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175`). At stages 2/3,
NEMO reconstructs `ww` from the Kmm velocity
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:324-337`) and calls
vector `dyn_adv` with Kmm velocity
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:472-480`). ZAD forms
the two-level vertical recurrence at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:100-130`; the state-changing
U/V subtractions are exactly lines 123-126.

## Unconditional ZAD verdict

The preserved candidate was measured at clean commit `71f9ba0a4fa4`.

| comparison | baseline | candidate | Rule-12 result |
|---|---:|---:|---|
| kt2 max U | 2.7478404751243857e-12 | 2.7377110452773967e-12 | cellwise FAIL |
| kt2 max V | 3.305560306813421e-12 | 3.284922138989399e-12 | cellwise FAIL |
| worsened rows by kt | Round-44: 57 | kt2:2; kt3:6; kt4-10:7 each = 57 | unchanged |
| first-over-bar | kt2 U/V | kt2 U/V | did not move later |
| worst row | - | kt9 V: 4.08449440707562e-6 = 18,394,927,489.702637 row ULP | FAIL |

The earliest-kt/largest row is kt2 V: 9,300 cells worsened; maximum worsening
is 2.3926184384431837e-13 = 1,077.53955078125 row ULP. Its handed-operand walk
finds stage-1 HPG `v_Kmm` already differs by 3.284922138989399e-12 (17,100
cells); stage-2/stage-3 HPG `v_Kmm` differ by 9.330750459097745e-6 and
9.61756891376675e-6. Thus HPG is the first kt2 consumer, at the compiled
`CALL dyn_hpg` (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-145`),
not the producer. The candidate's only producing change is the preceding
stage-program ZAD handoff, whose state-changing oracle statements are
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:123-126`. With NEMO's handed stage inputs, its stage-1/2/3
ZAD accumulator maxima are only 2.65e-23, 5.29e-23, and 4.24e-22: the failure
is live-input composition, not isolated ZAD arithmetic. The committed identity
bridge proves round-46 kt2/kt3 entries bit-identical to `year_owners/nemo_seed0`.

The 60-entry independent run likewise worsens immediate RMS U/V:
2.171950191174923e-13 -> 2.2003988870874693e-13 and
4.203385504352635e-13 -> 4.213442618393311e-13 at kt2. At kt60 the baseline ->
candidate RMS tuple (T,S,U,V,SSH) is `(5.259883e-3,3.823798e-4,7.637755e-4,
6.254840e-4,1.107934e-4)` -> `(5.259758e-3,3.823734e-4,7.637669e-4,
6.254633e-4,1.107707e-4)`; later aggregate improvement cannot legalize the
kt2 cellwise regression. Preregistration is REFUTED and ZAD was reverted.

| Rule-12 card | disposition |
|---|---|
| GYRE-zco | FAIL, 57 worsened rows; ZAD not landed |
| LOCK_EXCHANGE / OVERFLOW / DINO | NOT_ENTERED: conditional landing gate closed |
| ORCA2 | NOT_ENTERED; remains UNMEASURED_WITH_SPEC |

The conditional day-1..30 before/after run was therefore NOT_ENTERED; there is
no landed candidate to compare and no month claim.

## Default roots, provenance, and controls

The trajectory gate now defaults ROOT/STAGE2_ROOT/STAGE3_ROOT to
`round19_oracle_v2_external`, hashes each selected kt2 entry against that
canonical record, and records three VERIFIED identity rows. Its mismatch plant
is red. A default-argument kt1-10 replay compared all 70 rows with the explicit
v2 baseline: PASS, zero worsening, identical kt2 U/V first-over-bar. Focused
tests pass (28/28).

`year_fromrest/nemo_pristine/binary.sha256` records YRPERT hash
`578c88f...`, but adjacent `nemo` hashes `a759e8b...`, identical to the
R41ADVSP binary registered under `round41/oracle_dynadv_split`. Records were not
edited; this receipt records the truth.

The old round-46 gate's Round-48 VOR assertion was retracted in the tool: it now
retains the stage measurements as post-hoc and cannot abort on that superseded
owner prediction. Pre-code review found the conditional landing boundary and
v1-default hazard; post-code review found no retained physics delta, no NEMO or
record mutation, and no un-stamped measurement. The candidate/revert pair ends
at `5bfd0eb23f40`; retained commits before this receipt are `4715dd2f79be`
(roots), `0d5c15200afc` (identity bridge), and `5b1183376627` (retraction).
The citation gate passed all six compiled-source citations and its shifted-line
plant exited nonzero. The focused gate/citation/stamp suite passed 54/54; ruff
passed on every new or changed test/probe surface (the legacy full phase gate
still reports its pre-existing `surf_T`/`surf_S` naming findings).

## ASKED / UNASKED

| item | disposition |
|---|---|
| ZAD retest and conditional landing | ASKED; measured, REFUTED, reverted |
| scalar-v2 defaults/provenance truth | ASKED; fixed/recorded |
| month and other-card runs | ASKED only after clean ZAD; NOT_ENTERED |
| configuration, tolerance, state, NEMO edit/run | UNASKED/forbidden; none |

Open owner question: which prior live-input composition statement creates the
kt2 entry regression before stage-1 HPG; isolated ZAD arithmetic is at bar.
