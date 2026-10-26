# Preregistration: NEMO-testcases L2 GYRE round 65

Date: 2026-09-12

## Scope and immutable inputs

This round consumes the admitted round-64 `Krhs`/TKE oracle at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round64/oracle_krhs_split`
and the compiled GYRE source at
`cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo`.  It may change only the
shared legoESM transcription named by the first unequal compiled statement.
It does not change configuration, carried state, the year harness, the
reconciliation gate, the freshwater pair, or the #1484 guard.

## Calibration gate

Before scoring, `content_T`, `content_S`, `e3t_Kbb`, `e3t_Kmm`,
`diss_product`, `strat_product`, and `rhs_statement` must each report exactly
zero unequal cells against independent reconstruction.  The record must report
23 `Krhs` fields and 11 TKE fields, `kt=2`, stage 3, `Kbb=3`, `Kmm=2`,
`Krhs=1`, shape `(32, 22, 30)`, and fp64.  Any inequality stops the round and
names the compiled writer statement.  Truncation, one-ULP, and wrong-commit
plants must exit nonzero.

## `Krhs` boundary ladder

For T and S, score exact unequal-cell count and max absolute difference after:

1. zeroing `Krhs`;
2. the FCT upstream first guess;
3. the limited FCT anti-diffusive correction;
4. `tra_sbc`;
5. `tra_qsr`;
6. `tra_ldf`;
7. content assembly.

The first boundary with a nonzero exact count owns the walk.  A later exact
boundary does not erase an earlier inequality.

## Candidate and falsifier, preregistered before scoring

Candidate: the first non-bit statement is the upstream horizontal-flux
formation in compiled `traadv_fct.f90:503-506`.  NEMO passes the native staged
face products `pU=zFu` and `pV=zFv`, formed in compiled
`stprk3_stg.f90:295-296`; legoESM's FCT path instead reconstructs those products
from metric-free transports inside its divergence.  The candidate fix is to
let the one shared FCT implementation consume the native staged face products,
without changing values, time levels, configuration, or carried state.

CONFIRM only if the zero boundary is exact, the upstream-first-guess boundary
is the first unequal boundary, and substituting the recorded/native `zFu/zFv`
association makes that boundary bit-exact for both tracers.  REFUTE if an input
or an earlier upstream statement differs, if the first unequal boundary is
elsewhere, or if the native-product substitution leaves any unequal cell.  A
refutation lands no candidate patch.

Within FCT, score in compiled order: transport/thickness operands and time
levels; upstream face fluxes; first-guess divergence/update; centred fluxes;
anti-diffusive fluxes; nonosc local bounds; neighbour min/max; positive and
negative budgets; face limiters; final correction.  Substitute one recorded
operand or sub-result at a time; every association is reported separately.

## TKE RHS candidate

Score the recorded shear, stratification, and dissipation products before the
full RHS.  The preregistered candidate is the association of the dissipation
subterm `zfact3*dissl*en` into the parenthesised RHS in compiled
`zdftke.f90:428-431`, specifically line 430.  CONFIRM only if the three inputs
and individual products are exact and reproducing NEMO's statement association
makes all 238 reported cells bit-exact.  REFUTE if an operand/product is already
unequal or the association substitution leaves any unequal cell.

## Rule-12 cards and acceptance

If and only if a transcription is confirmed, land it in the shared production
operator and require: the GYRE `kt=1..10` ladder has no newly earlier
first-over-bar row; days 1--30 are scored against the recorded decision-36
before arm; LOCK_EXCHANGE and OVERFLOW resolved namelists determine whether the
statement executes and their executing rows are measured; DINO's separate
branch is measured with shared-statement risk explicit; ORCA2 remains
UNMEASURED only with an exact executable measurement specification.  Any moved
row is registered.  The round reports GYRE `kt=3` T/S and day-30 before/after.

No acceptance threshold is relaxed.  Exact discharge means zero unequal bits.
