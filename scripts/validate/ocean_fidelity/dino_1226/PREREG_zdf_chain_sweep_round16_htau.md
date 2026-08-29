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

Controls must be red: perturb one wet phase argument by +1 ULP and require
the glibc vector-SIN exact comparison to fail; roll the candidate latitude
field by one j-row and require the exact `htau` comparison to fail; retain the
existing row-18 one-cell roll, nonfinite, and planted-value controls.  No
downstream row is promoted by this discriminator alone.

Pre-measurement correction, 2026-08-28: the first version registered an
adjacent-i exchange.  That control is retracted because DINO's T-point
latitude is zonally constant, so exchanging adjacent i values cannot change
the input and cannot fail.  The j-row roll above is the red-capable replacement.
