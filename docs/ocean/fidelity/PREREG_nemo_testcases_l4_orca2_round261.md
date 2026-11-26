# ORCA2 round 261 preregistration — stage-2 V-transport operand split

Date: 2026-10-12. Frozen base: `0aee6bedb`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. The round starts at the
rank-0 kt=1 stage-2 `zFv` boundary admitted by round 260. No deck, card
selector, carried-state contract, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Compiled order and existing records

NEMO forms the stage-2 barotropic V correction in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279` and writes
`zFv` as `e1v * live_e3v * (vv(Kmm) + zvb * vmask)` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.
The source-ordered operands are therefore `e1v`, live `e3v`, associated
`vv(Kmm)`, `zvb`, and resolved `vmask`.

The round reuses the admitted round-222 OMT-4 records:

- `oracle_rkstage2_operands_kt00000001.bin` for NEMO's Kmm velocity and
  stage-2 velocity time levels;
- `oracle_bt_advmean_operands_kt00000001.bin` for the external-mode average;
- the stage frames for `r3v(Kmm)` and the associated stage-1 state; and
- `oracle_rktracer_operands_kt00000001_s2.bin` for the source-written `zFv`.

The existing round-232 source replay and the production D114 resolved-mesh
builder are reused. Any candidate value comes from the already-passive live
operand trace. No in-executable observer is allowed.

## Frozen predictions and falsifiers

1. **R261-P1 — provenance and reproduction.** Both labels run CPU
   fp64/libm production JIT from one clean committed tree. The round-260
   associated entry remains at the campaign floor, and rank-0 stage-2 `zFv`
   remains 849/8,100 unequal with maximum `0.7981690085143782 m3/s`.
   Failure to reproduce either row invalidates the split.
2. **R261-P2 — record and replay closure.** The admitted records are
   self-consistent at kt=1 stage 2, and replaying NEMO's five operands in the
   compiled multiplication/addition order reproduces recorded `zFv`
   bit-for-bit. Candidate operands likewise replay the passive candidate
   `zFv` bit-for-bit. Any endpoint mismatch makes the instrument invalid.
3. **R261-P3 — predicted first operand.** `e1v`, live `e3v`, associated
   `vv(Kmm)`, and resolved `vmask` are bit-exact on NEMO-live rank-0 support.
   The first valued non-bit operand is predicted to be `zvb`, inherited from
   the stage-2 external-mode average/correction. An earlier valued mismatch
   refutes this prediction and owns the walk in written order.
4. **R261-P4 — one-variable closure.** Substituting NEMO's first non-bit
   operand alone, while leaving the other four candidate operands fixed,
   reduces `zFv` to the campaign floor. If it does not, substitutions are
   accumulated in written order until the first closing cancelling unit is
   named; no partial member lands.
5. **R261-P5 — tracer consequence.** A closing stage-2 transport arm reduces
   the remaining fold-band stage-2 T/S errors (`6.435500132795369e-9 K` and
   `1.3842782919937235e-8 PSU`) to the `2e-10` floor under both labels. Failure
   means the next source-ordered tracer statement remains and the candidate
   is held.
6. **R261-P6 — terminal predicate.** Only a unit that closes P4-P5 is
   integrated into the complete private unit. It must remove the post-floor
   greater-than-10x boundary and complete OMT-4 kt=8..10 under both labels
   before Decision 96 landing gates are reached.
7. **R261-P7 — controls.** Exact worktree, both-label coverage, source order,
   record identity, candidate/oracle replay closure, round-260 reproduction,
   first-owner classification, substitution closure, and one-bit sensitivity
   each have a planted refusal.

## Stop conditions

No NEMO acquisition is planned. A genuinely absent required operand is
`STOPPED_FOR_RECORD`; a non-closing split or later numerical boundary is
`HELD`. No configuration choice, stabiliser, floor, clip, or partial
cancelling operand may land.
