# SI3 round 13 receipt — O7 retained-record admission and score

## Outcome

O7 is **READY TO LAND**. The retained, unmodified production
`ORCA2_SAS_ICE` run was re-admitted without rebuilding or rerunning NEMO.
At `kt=1,2`, all 91 scored leaves per step are 0 ULP on wet cells, full
arrays, wet seam cells, and wet fold cells under CPU fp64 production JIT and
the glibc/libm transcendental arm. Both run logs have `STOP 0`, both
`RUN_COMPLETE` markers are present, and the two reference/record restart
files are byte-identical.

Evidence is retained at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round13/refused/oracle_o7_orca2_sas_ice_20261025T172428Z_rc2_3061284`.
The successful `o7_admission.json` SHA-256 is
`67e799dab877deeb38aeafb15949c4de1b630720e6d5fea986c402210e5a34ac`.

## Refusal localisation and one-statement fix

The refusal's `full=1` was the **maximum signed-zero ULP distance**, not a
count of one array word. Before the fix, both `kt=1` and `kt=2` had 10,207
bit-different `qsb_ice_bot` words, all dry interior cells and all exactly
our `+0.0` (`0x0000000000000000`) versus NEMO's `-0.0`
(`0x8000000000000000`). There were zero wet and zero halo differences; 311
were in the scorer's seam columns, 335 in its fold rows, and 12 in both.
The first word was NumPy `(0,0)`, NEMO `(i,j,rank)=(3,3,0)`: dry, interior,
seam, not fold. Its recorded inputs included `smask0=at_i=at_i_b=vt_i=0`,
`sst_m=0`, and `t_bo=273.15`.

The first and only owner is the record-build second assignment
`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icesbc.f90:447`. It evaluates
`MIN(+0,-0)` and the emitted `MINSD` keeps its right-hand cap operand on the
tie; the following `zswitch *` therefore writes `-0`. This is a direct
interior calculation, not initialisation, halo fill, or `lbc_lnk`. The model
had the same operands but its generic helper kept the left-hand operand on a
tie. Only this cited call site changed, to the right-hand-on-tie comparison at
`packages/ice/legoesm/ice/si3_surface.py:510`.

The production `si3_surface_step` test at
`tests/ice/unit/test_si3_surface.py:615` failed before the change with
`0x0000000000000000` and passes after it with `0x8000000000000000`. The
independent scalar transcription was corrected to the observed compiler rule.

## Full O7 score

For every name below, the score tuple is identical at `kt=1` and `kt=2`:
`(wet ULP, full ULP, wet-seam ULP, wet-fold ULP, wet normalised) =
(0, 0, 0, 0, 0.0)`.

- `blk_ice_1` outputs/intermediate (9): `wndm_ice`, `zsipt`, `Cd_ice`,
  `Ch_ice`, `Ce_ice`, `theta_zu_i`, `q_zu_i`, `utau_ice`, `vtau_ice`.
- Post-exchange full-halo outputs (2): `utau_ice`, `vtau_ice`.
- `ice_alb` output (1): `alb_ice`.
- `blk_ice_2` outputs and statement intermediates (30): `qsr_ice`,
  `qns_ice`, `dqns_ice`, `qla_ice`, `dqla_ice`, `evap_ice`, `devap_ice`,
  `tprecip`, `sprecip`, `emp_oce`, `emp_ice`, `emp_tot`, `qemp_oce`,
  `qemp_ice`, `qns_tot`, `qsr_tot`, `qprec_ice`, `qevap_ice`,
  `qtr_ice_top`, `zcptrain`, `zcptsnw`, `zcptn`, `zsq`, `zsipt`, `z_qlw`,
  `z_dqlw`, `z_qsb`, `z_dqsb`, `zevap`, `zsnw`.
- `ice_flx_other` outputs plus retained input (4): `qsb_ice_bot`, `fhld`,
  `qlead`, `frq_m`.
- Independently composed `si3_surface_step` leaves (45): the same 9
  `blk_ice_1` leaves, 2 post-exchange stresses, albedo, all 30 `blk_ice_2`
  leaves, and the 3 `ice_flx_other` outputs. Composition therefore also
  proves the internal feed-forward values, rather than replaying stage
  products.

This is 46 direct stage comparisons plus 45 composed comparisons per step,
182 scored leaf-step pairs in total. The admission JSON retains the individual
five-number row for every leaf.

## Admission controls

Read-only preflight passed all 11 sections and 36 planted violations. The
retained admission then passed all 18 evidence/refusal plants: producer stamp,
binary/record/input checksum, source-derived time level, selector, error log,
restart, missing frame, wet one-ULP, swapped field, coefficient one-ULP, fold
sign, pond arm, snow-fraction selector, SUM seed, and composition feed.

The first successful score exposed a control-only cache defect: the wrong
native-transcendental arm had been checked only at `kt=1` after a cached libm
compile. Cache-cold measurement confirmed `kt=1` is inert, while `kt=2`
changes `alb_ice` by 2 ULP and composed shortwave leaves by up to 4 ULP. The
control now clears JAX caches and checks both preregistered steps; admission
records `FIRED_KT_2`. No score definition or bar changed.

## Invariance and gates

- Lane-3 C1D bulk gate: byte-identical to round 11's JSON (SHA-256
  `c5f96fb239913515ad3cb3ecf8b0e8917e0cd30edeb7d2dece4d0de4ba06ee03`),
  227,760 comparisons, 227,742 bit-identical, the same 18 native-exp rows;
  `qsb_ice_bot` is 8,760/8,760 bit-identical.
- D5 C1D year under CPU production JIT: the complete `stage_d_year_rescore`
  object is identical to the round-11 baseline; the landed
  `r6_step1_plus_step2_snow_update` full-year SHA-256 remains
  `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`.
- O1/O2/O3/O5 regression suites: 57 passed, 1 intentionally deselected.
- The two requested external reviewer CLIs could not run in this restricted
  sandbox: Claude failed DNS lookup and Argon was denied its localhost socket.
  This is an environment limitation, not positive review evidence.
- Per-receipt citation gate (all SI3 receipts, full map), default gate, and
  private-import ratchet: run from the clean landed commit below.

## CONFIRMED vs PLAUSIBLE

- **CONFIRMED:** retained reference and record runs completed cleanly;
  same-program/passivity manifests passed; all recorded O7 outputs and
  intermediates are 0 ULP at both steps on wet/full/seam/fold scoring; all
  admission plants fire; the refusal was a compiler-observable signed-zero
  tie at one cited statement and the production-call-site test is non-vacuous.
- **PLAUSIBLE:** none in the O7 score.

## Choices

ASKED: production deck, one rank, `kt=1..2`, CPU fp64 production JIT,
glibc/libm, exact wet/full scoring, signed-zero strictness, and the retained
`--admit-existing` route.

UNASKED: none. The native control repair implements the already preregistered
wrong-arm refusal over both preregistered steps; it does not change physics,
configuration, scoring, or bars.
