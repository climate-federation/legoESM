# Preregistration — round 165, developed TKE statement walk

Committed before admitting or scoring the Round-164 developed TKE records.
The entry is NEMO's admitted day-180 restart at step 1080 and the scored
program is legoESM's production-jitted step 1081.  Eager and isolated rows are
diagnostics only.

## A. Admit the existing acquisition

The operator's Round-164 NEMO run reached `STOP 0`, and its step-1080 restart
already passed the byte-for-byte comparison with the uninstrumented Round-132
restart.  Admission then refused on the first record magic check.  The compiled
writers, read before this preregistration, declare fixed-width 16-byte Fortran
characters whose source literals are 15 bytes and therefore carry one trailing
space:

- `GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r54_tke.f90:26,128-130`
  writes `NEMO_L2_R56TKE2 ` and the 13-integer operand header;
- `GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:24,65-67`
  writes `NEMO_L2_R101TKE ` and the 13-integer statement header.

The run script incorrectly compares each 16-byte slice with the unpadded
15-byte Python literal.  The one-variable repair is to compare with the padded
16-byte literals and to validate record stamps against the immutable producer
commit in `producer_commit.txt`, rather than against this later reader commit.
No rebuild or rerun is allowed.  Prediction: both registered sizes, both
headers (`kt=1081`, `Kbb=Kmm=3`, `32x22x31`, binary64), both SHA-256 stamps,
and the restart twin pass.  REFUTED if any of those checks fails; a partial
record is then not reused.

The admission plant changes one expected magic byte and must exit nonzero with
a named `REFUSE` line.  This directly controls the repaired comparison.

## B. Production-JIT compiled-order walk

Extend the existing developed-state walk and existing live TKE statement trace;
do not add a second bridge or an isolated replacement closure.  Score, in this
order, over every consumed wet cell:

1. entry `en`;
2. surface/bottom boundary result;
3. post-Langmuir `en`;
4. matrix upper/lower/diagonal and RHS;
5. post-sweep `en`;
6. raw and bounded mixing lengths;
7. `avm_k`, `avt_k`, `dissl` and the `zdfphy` copy.

The compiled statements are
`zdftke.f90:267` (entry), `:283-323` (boundaries), `:325-394`
(Langmuir), `:402-442` (Prandtl, matrix and shear/buoyancy/dissipation RHS),
`:475-494` (three recurrences and floor), `:611-714` (mixing lengths and
coefficients), and `zdfphy.f90:335-354` (closure call and coefficient copy),
all in the Round-164 compiled build above.

Frozen prediction: entry, boundary and post-Langmuir rows are BIT; the first
non-bit statement boundary is the RHS at `zdftke.f90:438-441`, and its first
non-bit consumed operand is the inherited shear production `p_sh2`, not the
RHS association itself.  REFUTED if any earlier row is non-bit, if the RHS is
BIT, or if an earlier RHS operand differs.  All failed predictions remain in
the receipt.

A one-ULP change to a nonzero consumed entry-TKE cell must move at least one
registered production-JIT boundary and exit nonzero with
`STATUS PLANT-FIRED`.  A record-field plant must likewise make admission or a
scored row fail; controls do not perturb zero or an unconsumed sentinel.

## C. Candidate and verdict

Only a one-variable statement whose own arithmetic is non-bit given NEMO's
recorded operands may become a candidate.  An inherited non-bit operand names
its upstream boundary and remains HELD; no downstream rewrite may conceal it.
Any candidate must pass the full Decision 43/45 gate as amended by Decision
55, including month/year rows and every executing card.  No configuration or
carried-state choice is authorized.  If the first owner needs another oracle
operand, status is STOPPED_FOR_RECORD with a fail-closed acquisition script;
otherwise this is a HELD measurement round unless the full landing gate passes.

The immutable production headline remains round 163's after arm: kt2 T/S/U/V
AT-BAR, first-over-bar kt3, day-30 T rms `6.572574374770603e-05` K, day-240
`1.644836070117868e-02` K, and day-360 `1.122566001855131e-02` K.
