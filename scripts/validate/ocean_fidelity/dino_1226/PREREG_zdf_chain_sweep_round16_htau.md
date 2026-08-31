# ZDF chain sweep round 16 — row-18 vector-SIN discriminator

Date frozen: 2026-08-28, after the direct `gdepw`/`htau` substitution and
before evaluating the oracle's linked vector sine.

The direct operand receipt localizes row 18 to `htau`: substituting `gdepw`
alone leaves 21/9,920 full-row failures, while substituting `htau` alone makes
the full row 0/9,920.  Both independent reviewers rejected the first proposed
fix.  NEMO's source association `rpi/180*gphit` and JAX's `deg2rad(gphit)`
produce the same binary64 phase, so changing only that spelling cannot clear
the dump.  The active NEMO binary calls glibc 2.34
`_ZGVbN2v_sin@GLIBC_2.22` in the vectorized `htau` loop.

Before production code changes, compile the committed two-lane ABI wrapper
and evaluate the captured DINO T-point latitudes in row-major, i-fastest
pairs.  Record glibc version and SHA256 for source, wrapper library, NEMO
binary, direct `htau` dump, and row-18 operand artifact.  Score:

1. exact binary64 element census of the source-ordered phase
   `(float64(pi)/180)*gphit` against the legacy JAX phase;
2. exact binary64 element census of glibc vector SIN against JAX/XLA SIN;
3. exact binary64 census and the unchanged `1e-15` per-column bar for
   `max(0.5,min(30,45*abs(vector_sin(phase))))` against the direct NEMO
   `htau` dump on all 9,920 wet columns and the four registered southern
   focus columns;
4. the complete literal row-18 update using that candidate `htau` against
   dumped post-row-18 `en`.

`CONFIRM-VECTOR-SIN` requires candidate `htau` to be bit-exact on every wet
element and the complete row to score 0/9,920, with all four focus columns
passing.  Any candidate `htau` bit mismatch or any complete-row failure is
`REFUTE-VECTOR-SIN`; no vector-SIN production option is then allowed.

Controls must be red: locate the first wet phase for which a +1 ULP change
changes the rounded glibc vector-SIN output, perturb that one argument only,
and require the exact comparison to fail once; roll the candidate latitude
field by one j-row and require the exact `htau` comparison to fail; retain the
existing row-18 one-cell roll, nonfinite, and planted-value controls.  No
downstream row is promoted by this discriminator alone.

Pre-measurement correction, 2026-08-28: the first version registered an
adjacent-i exchange.  That control is retracted because DINO's T-point
latitude is zonally constant, so exchanging adjacent i values cannot change
the input and cannot fail.  The j-row roll above is the red-capable replacement.

Pre-measurement correction 2, 2026-08-28: the first-wet phase itself is also
an insensitive +1 ULP plant—the correctly rounded vector sine does not change
there.  The aborted probe emitted no result.  An all-wet sensitivity census
found 8,046 wet phases whose +1 ULP plant changes the vector result; the
control now deterministically selects the first of those, then perturbs only
that one argument.  This preserves the registered one-argument/one-ULP test
without pretending an insensitive plant is red-capable.

Pre-measurement extension, 2026-08-28, after `REFUTE-VECTOR-SIN`: the vector
candidate reduced the direct `htau` exact miss from 74,005 to 3,364 wet
elements and cleared the complete row at its bar, but it failed the exact
operand requirement.  No production code was changed.  Disassembly shows the
oracle loads native degree-valued `gphit`, evaluates runtime `rpi/180`, then
executes `mulpd -> _ZGVbN2v_sin -> andpd(abs) -> mulpd(45) -> minpd(30) ->
maxpd(0.5)`.  legoESM's captured latitude has passed through its radian grid
representation before returning to degrees.  The existing `mesh_mask.nc`
already carries NEMO's direct `gphit`, so no new oracle dump is required.

Add the registered 2x2 substitution now: legacy captured latitude versus
direct dumped `gphit`, each with JAX/XLA sine and host `_ZGVbN2v_sin`.  Score
the direct latitude in degrees and the multiplied phase bitwise, then score
each constructed `htau` bitwise and through the complete row.  `LATITUDE`
owns only if direct-`gphit`+JAX is exact and legacy+vector is not; `VECTOR-SIN`
owns only for the converse; `INTERACTION` owns if only direct-`gphit`+vector
is exact.  If even the combined arm is not exact, stop and localize the next
assembly operation.  The same focus bars and red controls apply.
