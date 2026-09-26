# Preregistration: first-substep freshwater forcing discriminator, round 7

Date: 2026-08-29. Frozen before measurement.

## Registered discriminator

Round 6 held the owned U/V slow forcing exactly.  It left the earliest
downstream output, first-substep SSH, at `E=6.2048763005e-7`, followed by U/V
at `6.2120556998e-7` / `1.1172636476e-6`; exact holding of the BEFORE seed did
not change those values.  The next source-ordered operand is therefore the SSH
source in the continuity update.

NEMO computes
`ssha_e = sshn_e - rDt_e * (ssh_frc + zhdiv)` at active
`cfgs/DINO/MY_SRC/dynspg_ts.F90:718-725`.  legoESM computes
`eta_new = eta_c - dt_s*div_flux + dt_s*F_slow_eta` in
`barotropic_latlon_cgrid.py:805-868`.  The registered oracle substitution is
therefore **`F_slow_eta = -spg_dump_ssh_frc.bin`**, not the same-sign dump.
The existing dump is full-domain/haloless after stripping `nn_hls=2`, at the
same NOW forcing time level as `zu_frc/zv_frc`.

The round-7 arm holds all already disposed upstream inputs: exact U/V slow
forcing, exact BEFORE barotropic seed, and the signed SSH forcing above.  It
measures rows 1.3 and 1.4 with the unchanged ACCUMULATING `1e-12` campaign
bar, correlation minimum `1-1e-9`, and mean-absolute-ratio tolerance `1e-6`.

## Confirm / refute and ownership

- **CONFIRM SSH-source ownership:** first-substep SSH is `AT BAR` and its
  removal is at least 90%; U/V may either close or expose the next operand.
- **REFUTE:** first-substep SSH remains `DEBT`, or removal is below 90%.
- If SSH closes but U/V do not, `ssh_frc` owns the continuity component only;
  peel the next source-ordered surface-pressure-gradient/Coriolis/drag operand.
- Row 1.4 and rows 2--6 remain targeting-only until every earlier literal row
  is at bar.  No downstream score crosses the literal row-1.2 stop.

Admission retains round 6's clean CPU/fp64/lane/e3t gates, exact populations,
interception receipt, planted controls, and complete hashes.  The held signed
SSH field must compare exactly to `-ssh_frc` on all 9,920 wet T cells.  This is
an existing-dump offline arm; no NEMO writer or SLOT block is involved.

## Instrument amendment after invalid attempt 1

The first execution completed all three arms but the receipt rejected the
signed-SSH control because the registered NEMO dump is identically zero and
the generic normalized metric correctly refuses a zero reference RMS.  No
artifact was admitted.  The control is replaced with an exact-zero receipt:
9,920 wet cells, zero nonzero reference values, and zero maximum held-minus-
oracle difference.  The arm outputs, bars, and ownership rule are unchanged;
the zero dump itself preregisters a direct refutation of SSH-source ownership.

## Provenance retraction after adversarial review

The first nominally accepted round-7 receipt is **WITHDRAWN** with round 6:
its clean harness imported production modules from a different editable-install
worktree.  It must be rerun after the shared under-checkout import guard and
restoration assertions are committed.  No prior round-7 score is admissible.
