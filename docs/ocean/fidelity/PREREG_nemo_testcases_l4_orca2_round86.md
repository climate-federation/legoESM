# ORCA2 round 86 preregistration — resume the rung-0 frame debug build

Frozen before changing the round-85 launcher or requesting another build.
Base: `d67cbfd9ea8371461096ae057a842aeec371b996`.  Claim label:
**independent**.  This round repairs an acquisition preflight only; it changes
no package file, hierarchy assignment, shipped ORCA2 card, NEMO source, CPP
key, carried state, threshold, stabilizer, or sea-ice selector.

## Admitted inherited evidence

The operator ran the committed round-85 launcher.  It created
`ORCA2_OMIP_L4_R85FRAMEDEBUG`, copied the exact round-84 writer and patch, and
then exited 69 before `fcm build` with:

`REFUSE: debug build does not have one FCFLAGS assignment`

Inspection after that inherited failure shows the generated architecture has
one assignment at `arch_nemo.fcm:27` and one reference from the `%FFLAGS`
assignment at `arch_nemo.fcm:28`.  The launcher's unanchored
`grep -Fc '%FCFLAGS'` therefore returns two even though only one assignment
exists.  No executable, run directory, or debug trajectory was produced.
These observations are inherited/post-failure evidence, not a new crash
measurement.

The generated target is otherwise a resumable intermediate: its patched
`stprk3.F90` differs from the pinned source by exactly the four additions in
the committed round-84 patch, its writer module is byte-identical to the
committed module, and its architecture already contains the requested
`-O0 -g -fbacktrace -fcheck=bounds` flags.  The continuation must validate
those facts by hashes and a zero-fuzz reconstructed-patch comparison before
building; it may not regenerate, reinterpret, or modify the scientific deck.

## Frozen predictions and falsifiers

1. An assignment-anchored parser reports exactly one `%FCFLAGS` assignment on
   the inherited architecture while the retired literal-occurrence check
   reports two.  A count other than one/two respectively **REFUTES** the
   diagnosis.
2. The continuation accepts only the exact inherited intermediate: pinned
   architecture, build configuration, CPP keys, patched driver, frame module,
   original source, and absent binary/run output.  Any mismatch **REFUTES**
   safe resumption and must stop without rebuilding.
3. Two plants prove the corrected parser distinguishes an assignment from a
   reference: adding a second assignment must count two, and removing the only
   assignment while retaining the `%FFLAGS` reference must count zero.  A
   plant that stays green **REFUTES** the repair.
4. The resumed debug build uses the same round-84 patch and only changes
   compiler diagnostics.  Its resolved flags must contain
   `-O0 -g -fbacktrace -fcheck=bounds`, its preprocessed driver must contain
   four frame calls, and its binary must reject vector-math symbols.  Any miss
   **REFUTES** the build.
5. The carried round-85 crash prediction remains frozen: the run fails before
   stage 1 and its first source-resolved instrument frame is in the inherited
   surface canonicalizer, not the new frame writer.  A clean run, a completed
   stage 1, or a first frame in `l4_r84_frames.F90` **REFUTES** that
   attribution.
6. Debug output remains diagnostic-only.  No rung-0 state magnitude, ladder
   row, card result, restart identity, or month score is admitted from it.

## Stop condition

The sandbox may not launch MPI.  If the corrected fail-closed continuation is
the only deliverable, status is `STOPPED_FOR_RECORD` and the operator runs it.
Only a source-resolved backtrace can choose the subsequent instrumentation
repair; no legoESM physics lands before a fresh optimized 80-frame record also
passes terminal-restart identity against round 83.

ASKED: repair the false FCFLAGS refusal, safely resume the exact generated
debug target, and obtain the source-resolved line required by round 85.

UNASKED: regenerate a different target, change any scientific configuration,
guess a frame repair, admit debug values, change the shipped card or its ice
tuple, or relax any gate.
