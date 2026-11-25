# ORCA2 round 335 preregistration — standalone S1 bulk acquisition

Frozen 2026-10-25 before the first round-335 model score.  Base:
`072e4aa1b1c8`.  This preregisters an operator-run acquisition but does not
build its tool, configure NEMO, or execute NEMO.

## Record identity

The future deck/build is `ORCA2_STANDALONE_IIF_OMIP_L4`, derived from the
round-5 record deck with `nn_ice=1`.  Whether `key_si3` remains compiled is a
separate unresolved build-identity decision; the design's pick is to retain
it so the ocean mixing/drag branches match the comparison card, subject to a
boot/ten-step smoke gate.  The selected ice-cover file is also unresolved and
must be filled only after both design decisions are answered.  S1 itself
precedes the ice call, so its numeric record is independent of ice-cover
values; the configuration identity is not.

Run two ranks (`jpni=2,jpnj=1`) for steps kt 1--10 from the same from-rest
state used by the standalone reference.  Produce independent A/B twins.  The
writer is write-only, opens a separate rank file, and records owned bounds,
global indices, dtype, field names, kt, rank, deck/build/source hashes,
namelist digest, executable digest, and input-file digests.

## Frozen boundaries and arrays

1. **S1-E / entry**, immediately after the S0-F temporal-input and S0-M
   two-step-surface-mean boundaries, wind rotation, and immediately before
   `blk_oce_1`: `wndi,wndj,tair,humi,qsr_down,qlw_down,precip_raw,snow_raw,
   slp,sst_m,ssu_m,ssv_m,smask` plus the resolved S1 deck scalars.
2. **S1-1 / post `blk_oce_1`** at its return:
   `theta_air,q_air,precip,tsk,ssq,Cd,Ch,Ce,rhoa,sensible,latent,evap,utau,
   vtau,taum,wndm,qsr`.
3. **S1-2 / post `blk_oce_2`** at its return: all S1-1 public fields plus
   `qlwn,qns,emp`.

Every array is stored as raw native binary64 in a fixed declared order.  The
record includes the full local allocation and owned bounds; scoring uses the
owned wet T cells and publishes each rank separately before any assembly.

## Frozen predictions and falsifiers

- **R335-S1-P1, phase-2h baseline.**  Before this round's literal path, the
  existing O1 gate reports unequal counts `117,3964,8794,8794,8794,8794,8794`
  of 8,794 wet cells for `theta_air,ssq,utau,vtau,sensible,latent,evap`.
  Any different parent-worktree replay refutes the baseline and stops scoring.
- **R335-S1-P2, literal improvement.**  The literal path must make no field's
  unequal count larger.  It lands only if every same-record S1 output is
  exact, or Decision 122/139 admits it with all required safeguards.  The
  first still-unequal field and cell are recomputed from scores, never copied
  from a marker.
- **R335-S1-P3, determinism.**  A/B record files are byte-identical per rank.
  A single unequal byte, missing step/rank, or changed production output
  refuses the record.
- **R335-S1-P4, controls.**  Header/name/order/truncation/digest/rank-swap and
  one-ULP wet payload plants all exit nonzero.  Inexact unit pins, branch
  plants, eager-vs-JIT, finite-difference gradients, and signed-zero tie tests
  must pass.  Every non-S1 caller of changed shared code has a real
  parent-worktree-vs-current output control; a self-comparison is invalid.

## Admission and stop rule

Admission first checks the exact deck, build, compiler flags, ranks, steps,
source/object/writer hashes, all input hashes, and twin identity.  It then
replays S1 from each entry frame under CPU fp64 plus scalar glibc
transcendentals and scores every output word.  A mismatch in an entry operand
stops at S1-E.  Otherwise the first source-ordered post-call mismatch owns the
next work; no later aggregate is used to explain it.

No acquisition script is authorized in round 335.  The operator can request
one after the ice-cover decision fixes the standalone deck identity.
