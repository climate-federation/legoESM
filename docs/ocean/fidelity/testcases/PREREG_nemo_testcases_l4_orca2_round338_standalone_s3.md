# ORCA2 round 338 preregistration — standalone S3 ice-interface acquisition

Frozen 2026-10-26.  Base: `aae42be88d67`.  This preregisters an operator-run
acquisition of NEMO's `sbc_ice_if` boundary.  It does not build the tool,
configure NEMO, or run NEMO.  The offline unit tests of this round are
self-checks of the transcription, not scores against a NEMO record.

## Prerequisites (in this order, all fail-closed)

1. The Decision-171 ice-cover file exists, produced by the operator task in
   `orca2_standalone_design.md` (S3, "Ice-cover file production"), with its
   digest recorded.  Its producer run is a pending user decision.
2. The K0 reference build `ORCA2_STANDALONE_IIF_OMIP_L4` (Decision 175:
   `key_si3`, `nn_ice=1`, `ln_drgice_imp=F`, `ln_mxl0=T`, `nn_mxlice=0`,
   `ln_icebergs=F`) exists, and its operator boot plus ten-step smoke run has
   passed the design's checklist.
3. The K0 build's own ppsrc is compared line by line with the record build's
   `sbcice_if.f90`, `sbcmod.f90` and the EOS-80 branch of `eosbn2.f90`.  Every
   call point below is cited from the record build's ppsrc
   (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/`) because no K0 build
   exists yet; any line shift is re-anchored, any statement change refuses.
   The K0 object code is re-audited for the MAXSD/MINSD operand order, the
   absence of fused multiply-add, and inline `sqrtsd`, as this round did on the
   record object.

## Record identity

Two ranks (`jpni=2, jpnj=1`), steps kt 1--10, from the K0 initial state, CPU
binary64, independent A/B twins.  The writer is write-only, sits outside every
NEMO loop, opens one file per rank, and stamps owned bounds, global offsets,
dtype, field order, kt, rank, deck/build/source/patch hashes, namelist digest,
executable digest and every input digest, including the ice-cover file.

## Frozen boundaries and arrays

Lines are record-build ppsrc lines.

1. **S3-E, entry**: in `sbcmod.f90` immediately before `CALL sbc_ice_if` at
   :500.  `qsr,qns` (interior `A2D(0)`); `sst_m,sss_m,fr_i,ff_f` and
   `tmask(:,:,1)` (full `jpi x jpj`); `ts(:,:,1,jp_tem,Nbb)` (full, including
   the halo); `smask0` (interior); the unchanged channels
   `emp,sfx,utau,vtau,taum,wndm`.  `Kbb = Kmm = Nbb` because
   `stprk3.f90:155` calls `sbc(kstp, Nbb, Nbb)`.
2. **S3-F, cover**: in `sbcice_if.f90` immediately after `fld_read` at :149:
   `sf_ice(1)%fnow(:,:,1)` (interior).  S0-F owns its construction; S3 only
   consumes it.
3. **S3-Z, freezing point**: immediately after :159: `fr_i` (full), the
   masked EOS-80 freezing point.
4. **S3-C, clamp**: immediately after the loop ending at :167:
   `ts(:,:,1,jp_tem,Kmm)` (full).
5. **S3-X, exit**: in `sbcmod.f90` immediately after the call at :500 returns:
   every S3-E array again.

S3-F/Z/C are written only when `MOD(kt-1, nn_fsbc) == 0` (kt 1,3,5,7,9 at
`nn_fsbc=2`).  Raw native binary64, fixed declared order, full local
allocation plus owned bounds.

## Frozen predictions and falsifiers

- **R338-S3-P1, exact replay.**  Replaying the round-338 literal routine from
  S3-E plus S3-F gives `0 / n` unequal words, per rank and SBC step, for
  `fr_i` and `ts` on the FULL array (halo included), and for `qsr,qns` on the
  interior, signed zeros included.  S3-Z and S3-C are scored first; the first
  unequal boundary in source order owns the next work.
- **R338-S3-P2, unchanged channels.**  `emp,sfx,utau,vtau,taum,wndm` are
  bitwise equal between S3-E and S3-X at every step.  Any change refutes the
  source reading and stops.
- **R338-S3-P3, no-op steps.**  At kt 2,4,6,8,10 every S3-X array is bitwise
  its S3-E value.
- **R338-S3-P4, land signed zero.**  On every cell where S3-E has
  `tmask = 0`, `sss_m = +0` and `ts = +0`, S3-X has `ts = -0` (EOS-80 gives
  `-0.0575 * (+0) = -0`; the masked product stays `-0`; MAXSD keeps its second
  operand on the tie).  One such cell exiting `+0` refutes the tie audit.
- **R338-S3-P5, inactive damping.**  Because the flux loop reads the
  clamped temperature, `MIN(0., ts - zt_fzp)` is the literal `0.` on every
  finite cell, so qns under observed and model ice is
  `((1-cover)*qns + cover*zqi)*smask0 + cover*smask0*(-0 + -0)`.  Replaying
  with the MIN term deleted must give identical bits; any difference refutes
  the `Kbb = Kmm` reading.
- **R338-S3-P6, controls.**  Header/name/order/truncation/digest/rank-swap,
  one-ULP wet payload, one-ULP halo `ts` and a time-level plant all exit
  nonzero.  The census counts real clamped cells (owned and halo separately)
  and real `fr_i = 1` cells; it never uses an unconditional counter.

## Admission and stop rule

Admission checks deck, build, flags, ranks, steps, source/object/writer
hashes, input hashes (ice-cover file included) and twin identity, then
replays under CPU binary64 JIT.  An S3-E operand mismatch against the S2/S1
record stops at S3-E.  No aggregate explains a cellwise mismatch.  No
acquisition script is authorized in round 338.
