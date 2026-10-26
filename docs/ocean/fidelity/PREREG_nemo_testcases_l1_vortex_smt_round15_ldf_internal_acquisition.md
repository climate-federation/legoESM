# Preregistration — VORTEX_SMT round 15 (lane round 227): internal LDF record

Frozen before changing or running the acquisition instrument.  Base: lane tip
`e052580833d6f55f165a9f9395bbe3bbc7687777` (round 226).  Evidence belongs
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round227/`.

## Compiled branch read first

The admitted SMT-3 build calls `eos` and then `ldf_slp` before the RK3 stages
(`stprk3.f90:172-178`).  Its stage-3 tracer call dispatches the resolved
Laplacian/isoneutral branch to `traldf_iso_lap` (`traldf.f90:105-110`).  That
compiled branch computes masked tracer gradients (`traldf_iso.f90:178-214`),
the horizontal rotation-matrix entries and fluxes (`:230-246`), the vertical
entries and flux (`:259-285`), and the final divergence into `Krhs`
(`:287-304`).  Before those loops, `traldf_iso_a33` forms the squared-slope
coefficient and MSC split (`:780-795`, `:811-830`).  The slope producer applies
its live partial-cell thickness bounds before the Shapiro-filtered U/V/W
slopes (`ldfslp.f90:203-257`, `:265-338`).

Round 226's existing self-describing stage-3 record has only the tracer RHS
before and after `tra_ldf`; it has none of these internal boundaries.  Its
post-LDF T error is `2.0915088416728622e-07 K`, but no statement inside the
operator is yet attributable.

## Predictions and falsifiers

* **R15-P1 — one record extension, no physics.**  A new SMT-3 instrumented
  target copies the admitted round-224 deck and cpp keys byte-for-byte.  Its
  only source changes are additive read-only dumps at completed compiled
  boundaries in `ldf_slp` and `traldf_iso_lap`; no NEMO expression, loop
  bound, time level, namelist value, or legoESM model file changes.  REFUTED
  by any removed source line or any deck/cpp difference.
* **R15-P2 — sufficient compiled-order seams.**  The self-describing record
  carries, by name, the `ldf_slp` inputs, raw/bounded gradients, mixed-layer
  operands, pre/post-filter U/V/W slopes; the A33/MSC inputs and outputs;
  tracer gradients; A11/A22/A13/A23 and horizontal fluxes; A31/A32 and
  vertical fluxes; and the divergence/RHS increment.  REFUTED if any named
  family is absent or duplicated, or if the parser does not reach EOF on a
  group boundary.
* **R15-P3 — passive writer.**  The instrumented step-10 restart is
  byte-identical to the plain SMT-3 restart.  Every stored difference boundary
  rebuilds exactly from its own stored operands where an in-run calibration is
  defined.  REFUTED by a restart difference, non-finite payload, or failed
  calibration; a differently instrumented stream is informational only under
  the developed-record admission rule.
* **R15-P4 — fail-closed format and provenance.**  The checker derives payload
  sizes from each group's own rank/extents, requires only the magic and named
  groups, and refuses header, field-name, and truncation plants with named
  nonzero exits.  The record stamp contains the committed producer SHA and
  record digest.  REFUTED by a zero-exit plant, a hand-predicted byte count or
  header tuple, or an unstamped record.
* **R15-P5 — acquisition boundary.**  This round performs patch applicability,
  additions-only checks, shell/Python checks and a `gfortran -fsyntax-only`
  proof, but does not run `mpirun` in the sandbox.  It stops for the operator's
  record and makes no internal-LDF or trajectory verdict.  REFUTED if the
  record already exists and admits, or if this round claims a scientific row
  without it.

## Frozen next measurement

After admission, the next round first reproduces round 226's stage-3 entry,
pre-LDF and post-LDF rows.  It then walks the new seams in the compiled order
above through the production-JIT closure.  The first seam non-bit given NEMO's
own preceding seam owns the walk; no downstream statement is a candidate
before that seam is explained.  Isolated eager/JIT values may accompany the
production row but cannot replace it.

## No hidden choices

No physical, numerical, configuration, carried-state, threshold, stabiliser,
or acceptance-bar choice is made.  A new target name is required only because
the writer differs from round 224's already-admitted binary.  The acquisition
uses NEMO's existing SMT-3 deck, ten-step run length, grid, timestep and
precision unchanged.
