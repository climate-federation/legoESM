# Preregistration — NEMO testcase L2 GYRE round 134

Date: 2026-09-20

Incoming lane tip: `91864f1d9e9c68f2c44ea7be918b8936ded8a4a1`

This document is frozen before the acquired Round-132/133 daily record is
admitted and before any Round-134 trajectory is run. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round134/`.

Round 129 proved that the year gap is systematic, Round 130 showed that the
held locally exact patches do not carry its day-240 magnitude, and Round 131
stopped because the then-certified oracle record ended at day 30. The operator
has now reported a successful Round-133 acquisition at
`phase3/round132/oracle_daily_restarts`. This round performs the already-frozen
daily-nudging attribution; it is an oracle-input sensitivity experiment, not a
production configuration or a landing candidate.

## P0 — fail-closed record readmission

Extend and rerun the existing Round-131 daily-record gate against exactly
`phase3/round132/oracle_daily_restarts` from a clean committed worktree. The
record must contain exactly the 360 completed-step restart boundaries
`kt=6,12,...,2160`, with no missing or duplicate step, the source-card cadence
`nn_stock=6` and `nn_itend>=2160`, and the fields consumed by the four arms:

* tracers: `tn`, `sn`;
* vectors: `un`, `vn`, `uu_n`, `vv_n`, `ub_e`, `vb_e`, `ubb_e`, `vbb_e`;
* free surface: `sshn`, `ssha`, `sshb_e`, `sshbb_e`; and
* TKE: `en`, `avm_k`, `avt_k`, `dissl`.

The `uu_n`/`vv_n` pair strengthens the earlier 16-field gate: the compiled
restart writer records these live momentum slots separately from `un`/`vn`, so
the vector arm is refused rather than silently leaving them free. Every 30-day
overlap must be bit-identical to the admitted monthly NEMO year record. The
ready marker, source hashes, variable dimensions, finite values, dtype, and
one time record per file are also mandatory.

Frozen prediction: admission passes for all 360 boundaries and all 18 fields,
including `uu_n`/`vv_n`; all shared monthly arrays are bit-identical. Any
missing boundary or field, schema or cadence mismatch, non-finite value, or
non-bit monthly twin falsifies the prediction and stops the round before a
model step. A required-field plant and a monthly one-ULP plant must print
`STATUS PLANT-FIRED` and exit nonzero.

## P1 — one production-path harness, four isolated resets

Extend the existing `nemo_testcase_l2_gyre_year_fromrest.py` harness rather
than writing another stepper. First run an unnudged daily control from the
incoming implementation and require its monthly snapshots to be bit-identical
to the immutable Round-130 free arm
`phase3/round130/arms/r109_handoff/lego_seed0_year`. Its registered reference
values are T3D RMS `6.89043148782590898e-05 K` at day 30,
`1.64467402331753935e-02 K` at day 240, and
`1.12235712478436639e-02 K` at day 360. A mismatch stops attribution.

Then run four independent member-0 arms for 2,160 steps with the certified
GYRE-zco card and fp64/libm policy. At every completed daily boundary, snapshot
the pre-reset state and replace exactly one family for the next step:

1. tracer: `tn/sn` -> model `T/S`;
2. vector: `un/vn`, `uu_n/vv_n`, and
   `ub_e/vb_e/ubb_e/vbb_e` -> model `u/v`, the live momentum slots, and the
   four velocity histories;
3. free surface: `sshn`, `ssha`, and `sshb_e/sshbb_e` -> model surface state,
   the next stage-1 pre-solve free-surface operand, and the two surface
   histories; and
4. TKE: `en/avm_k/avt_k/dissl` -> model TKE state and coefficients using the
   existing restart bridge.

All non-member leaves remain free, and the reset occurs only after the day's
score snapshot. The `ssha` value is supplied through the production step's
existing stage-1 W/ZAD operand seam for the one step that consumes it; it does
not add a production carried-state slot or change the default path. The
diagnostic seam must be dynamic under the full production JIT, default-off
bit-identical, and covered by a production-path plant. Day 1 must therefore be
bit-identical across the control and all four arms; any moved day-1 field
falsifies reset timing.

The harness must fail closed on a dirty or wrong commit, wrong record hash,
missing daily file, shape/dimension mismatch, absent live state, or a reset
registry that does not contain exactly the declared family. A planted one-ULP
source change must be detected and exit nonzero.

## P2 — frozen attribution and falsifiers

At days `30,60,90,120,180,240,300,360`, register for every family and each of
`T`, `S`, `u`, `v`, and `ssh`:

* free-versus-NEMO RMS;
* reset-versus-NEMO RMS;
* reset-versus-free RMS; and
* removed gap = free-versus-NEMO minus reset-versus-NEMO.

The owner family is the arm with the largest positive removed day-240 T3D RMS;
a non-positive removal exonerates that family under this intervention. Carry
forward Round 131's frozen directional prediction: the vector/history arm wins
and the first reset-induced T difference is born after the day-1 boundary in
the western third and upper 100 m. A different winning family, a different
birth region/depth, or no positive removal refutes that prediction and remains
reported as `REFUTED`; it is not repaired post hoc. Report the first differing
daily boundary and the Round-122 region/depth partition for the actual winner.

Only six-step daily boundaries exist, so the requested three-step cadence is
not run and no interpolation is permitted. Direct tracer reset can measure
control leverage without establishing the first causal statement; the receipt
will distinguish reset leverage from statement ownership.

## P3 — outcome and scope

All four admitted arms must complete before ranking. This round lands no
physics, changes no card or production carried state, and makes no
configuration choice. Its expected status is `HELD` with the attribution table
and an OPEN item that turns the winning family into the next magnitude-ranked,
compiled-order experiment.

A separate read-only Codex pass must try to refute record completeness,
boundary timing, family isolation, production-JIT use, the `ssha` mapping,
scoring, plants, and the ownership language. A `DO NOT SHIP` verdict blocks the
receipt. Every compiled-source statement used in the receipt must be mapped by
the citation gate; its shifted-citation plant must exit nonzero. Focused tests
and the full ocean fidelity/unit trees remain required, with every summary line
and any pre-existing-red comparison recorded.
