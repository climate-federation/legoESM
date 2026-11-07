# ORCA2 round 85 preregistration — rung-0 frame-writer crash diagnosis

Frozen before requesting any debug build or rerun.  Base:
`ac0205fe270d87613baf53697fa9bb29376626cc`.  Claim label:
**independent**.  This round diagnoses an acquisition-only crash; it changes no
package file, rung-0 scientific assignment, shipped ORCA2 card, NEMO source,
CPP key, carried state, threshold, stabilizer, or sea-ice selector.

## Admitted inherited evidence

The operator's round-84 optimized frames build compiled and entered `kstp=1`,
then exited 139 before stage 1.  Both ranks wrote a stage-0 frame of 14,288,244
bytes.  The inherited backtrace has addresses but no source lines because that
binary was built at `-O3` without `-g`.  Its address-to-symbol mapping was
inspected before this preregistration and is therefore explicitly **POST-HOC**,
not confirmatory evidence: the top NEMO symbol is the pre-existing
`l4_canon_2d` helper, called from the pre-existing surface-input recorder, not
the new `r84_dump_frame` symbol.

The compiled driver calls the new stage-0 writer before I/O/calendar and calls
the inherited surface-input writer after `sbc`
(`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:107-154`).  The latter
canonicalizes many surface arrays through `l4_canon_2d`; its only dereference
of the supplied field is the wet-cell assignment
(`ORCA2_OMIP_L4/MY_SRC/l4_oracle_canon_subroutines.h90:18-27`).  Round 83's
admitted binary contains neither canonicalizer symbol nor surface-input file,
so successful round-83 execution does not certify the currently accumulated
`MY_SRC/stprk3.F90` instrumentation.

## Frozen predictions and falsifiers

1. A fresh debug build applies the exact committed round-84 module and patch
   to the same pinned rung-0 deck, with `-O0 -g -fbacktrace -fcheck=bounds` in
   addition to the scalar-math flags.  The resolved build log must print all
   three required debug flags and the same four `r84_dump_frame` call sites.
   A different patch, deck, CPP key set, or missing flag **REFUTES** the run.
2. The debug run reproduces before stage 1.  Its first source-resolved NEMO
   frame is predicted to be the inherited `l4_canon_2d` wet-cell assignment,
   called by `l4_dump_ocean_surface_input`; the debug run must name a source
   line.  A first frame in `l4_r84_frames.F90`, a different NEMO routine, a
   completed stage 1, or a clean exit **REFUTES** this attribution.
3. Both inherited stage-0 files parse to physical EOF under the committed
   self-describing frame reader, with five finite fields and the expected
   rank/step/stage headers.  Any malformed or non-finite payload **REFUTES**
   the claim that the new writer completed its entry calls before the crash.
4. The debug binary and its output are diagnostic only and can never satisfy
   the frame-record admission gate.  No state magnitude, restart comparison,
   card result, ladder row, or month score is taken from it.
5. No acquisition patch is repaired until the source-resolved backtrace names
   the failing statement.  The eventual repair must be additions-only against
   the admitted round-83 record build and must run under a fresh production
   target; all 20 rank-step terminal restarts must be byte-identical to round
   83 before any of its 80 frames are admitted.  One unequal byte, one missing
   frame, or one green corruption plant **REFUTES** the repair.

## Stop condition

The sandbox cannot launch MPI under the standing PMIx rule.  If only the debug
launcher can be prepared this round, the status is `STOPPED_FOR_RECORD` and the
launcher is the sole acquisition request.  No frame fix is guessed from the
optimized backtrace, and no legoESM physics lands before the production frames
record admits.

ASKED: diagnose the round-84 frame acquisition crash with a source-resolved
debug reproduction, then repair and reacquire without changing model physics.

UNASKED: using the debug trajectory as evidence, changing any hierarchy rung,
editing NEMO's canonical source, changing the shipped card or its sea-ice debt
tuple, adding a stabilizer, or relaxing any gate.
