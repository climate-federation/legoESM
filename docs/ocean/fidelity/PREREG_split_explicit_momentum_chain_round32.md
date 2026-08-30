# Preregistration: row-1.4 substep trajectory, round 32

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. This measurement is **HELD** and
frozen before a NEMO build or run. Round 31's source-ordered momentum-commit
candidate was refuted: the CPU/fp64 production replay at SHA-256
`7ce94a68528e0e1b0fe950eeaa1ca194bcd9bbdfa0c8545459bbd7141888643d`
is byte-for-byte metric-identical to round 28. The selector was removed.

Scorer-path amendment, frozen after the build/run/bracket but before any
trajectory score: the retained artifacts prove the faithful card executes
`differentiable_barotropic=False`, hence the production outer loop is
`fori_loop`, not `scan`. They also prove trace row 1 is bit-identical to the
existing loop-entry dumps and not the post-substep-1 dumps. The scorer must
therefore tee the 68-row `fori_loop` into a diagnostic scan and return each
carry **before** applying that row's body. No metric, bar, population, input,
NEMO byte, or disposition changes; this corrects only temporal alignment and
the capture mechanism before the first score.

Control amendment, frozen after a dry scorer returned `INVALID` and before an
admissible score: round 6's printed “substep 1” is a separate one-row replay
with `barotropic_time_filter="nemo_ab3am4"`; it is not the first update of the
full 68-row `nemo_boxcar_ab3` recurrence. It remains a valid registered proxy
but cannot control this trajectory. Instead, NEMO trace row 1 must be
bit-identical to all three existing loop-entry dumps, trace row 2 must be
bit-identical to all three existing post-substep-1 dumps, and production trace
row 2 must pass the unchanged strict bar. The dry score also proved that four
nextafter steps at the maximum-magnitude entry SSH produce only
`0.971e-15` maximum/NEMO-RMS and therefore cannot cross the `1e-15` bar. Five
steps is the minimal deterministic firing plant. The failed four-step receipt
is retained; the required control is the five-step plant. No science row or
disposition from the invalid dry score is admitted.

## Instrument and source

Patch the cumulative 223-stream round-26 writer source with
`nemo_spg_row14_substep_trace.patch`. It adds exactly three deterministic
full-halo, zero-initialized streams at the pre-update site: `sshn_e`, `un_e`,
and `vn_e` at the start of every `jn=1..icycle`. The prior row's end swap
commits the next row's start; row 1 is the loop-entry state. Units 9440--9442
are a documented contiguous range and must be
unclaimed in the complete pre-patch tree and unique afterward. The OFF/ON
bracket is 223/223 exact plus those three additions; expected file size is
`icycle*jpi*jpj*8 = 6,184,192` bytes for DINO's 68x203x56 trace.

The scorer captures the corresponding production carry without changing model
code: a temporary wrapper converts only the 68-row outer `lax.fori_loop` to a
diagnostic scan, returns carry-in `(eta,U,V)` as `ys`, restores the original
`fori_loop` in a `finally`, and verifies one capture for each of round 6's two
arms. NEMO native U/V
are compared to production `U[:,1:]` / `V[1:,:]`; all trace files remain
full-halo on disk and are cropped by the cited A2D loader.

## Bars and dispositions

At every substep-start report the canonical tuple on the unchanged 9,920/9,758/9,868
wet populations. Apply the strict unconditioned 1e-15 pointwise class bar as a
diagnostic in addition to the registered field gate. Identity must be zero, a
five-nextafter plant must fail the strict bar, NEMO trace rows 1/2 must bind
the existing entry/post-substep-1 dumps exactly, the actual full-loop
production row 2 must be AT BAR, trace row 68 must be finite, tracked state must stay
clean, and all binaries, sources, streams, bracket, and prior artifacts are
SHA-bound.

* **SSH_PROPAGATION_LOCALIZED:** U or V is the first strict failure and SSH
  first fails at a later named substep. Record that first SSH substep as the
  next operand-instrument site; no QCO change is admissible.
* **SSH_FAILS_WITH_VELOCITY:** SSH and a velocity first fail at the same
  substep. Instrument that substep's continuity flux operands next.
* **NO_STRICT_SSH_FAILURE:** all 68 SSH rows remain at the strict bar; the
  final filtered accumulation, not the recurrence state, is the next owner.
* **INVALID:** bracket, restoration, shape, population, bound-substep, or
  planted controls fail.

This trajectory does not release row 3. Rows 4--6 and later chains remain
ordered-blocked until the inherited row-1.4 residue is either removed at its
source or accepted by an existing registered gate without relaxing a bar.
