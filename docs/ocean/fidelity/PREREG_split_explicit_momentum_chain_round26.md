# Preregistration: row-1.3 bottom stress and vector update, round 26

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before building or running the
held instrument. Bound predecessor: round-25 artifact SHA-256
`a9db28c458bb6e409ec23b74ea371b2d2e4aeaf1d84de322074ab72b46fca16e`.

## Ordered question and existing-dump inventory

Round 25 closes the last live-EEN coefficient association exactly, but the
faithful literal coefficient builder remains designed, not built. The next
executed source row is explicit bottom stress at
`dynspg_ts.F90:700-705`, followed by DINO's active vector update at
`:719-732` (`ll_wd=.false.`, `ln_dynadv_vec=.true.`).

The retained round-22 run already contains the substep-1 inputs
`cor2d_dump_ua_e_in_substep1.bin`/`va_e` and the pure-Coriolis outputs
`cor2d_dump_zu_trd_substep1.bin`/`zv_trd`, plus frozen slow forcing and the
QCO/PGF operands. It does not contain the executed trend after bottom stress,
the live `zCdU` face coefficients and reciprocal depths, or `ua_e/va_e`
immediately after the vector update and before halo/boundary processing. An
offline reconstruction cannot replace those missing executed operands under
the ordered-row rule.

## Held measurements

One write-only patch adds twelve full-halo binary64 streams at substep one:

1. `zCdU_u`, `zCdU_v`, `hu_e`, `hv_e`, `hur_e`, and `hvr_e` immediately
   before the bottom-stress expression;
2. `zu_trd`, `zv_trd` immediately after explicit bottom stress;
3. `zu_spg`, `zv_spg` used by the update; and
4. `ua_e`, `va_e` immediately after the active vector-form update and before
   wetting/drying, lateral-boundary, AGRIF, or swap processing.

The patch owns units `9420--9431`. The build gate inventories the entire
cumulative `MY_SRC` tree: every unit must have zero numeric references before
the patch and exactly one `OPEN` owner afterward. This range is reserved for
the row-1.3 bottom/update writer in future handoffs.

The OFF binary contains the certified deterministic-writer stack plus the QCO
continuity and EEN-coefficient patches. The ON binary differs only by the new
patch. The exact bracket requires 211/211 shared streams byte-identical and
exactly twelve new ON streams of 90,944 bytes. Missing-stream and one-bit
plants must fire. All handoff substitutions use the canonical
`__MEASURED_<name>__` SLOT prefix.

## Frozen score bars

The first score binds the exact bracket and the round-25 SHA. It first checks
NEMO's own source identities on the 9,758 U and 9,868 V wet-face populations:

- `(after_bottom - coriolis_before) == (zCdU * velocity) * reciprocal_depth`;
- `reciprocal_depth == mask / (face_depth + 1 - mask)`; and
- `updated_velocity == (velocity + rDt_e * (pgf + after_bottom + forcing)) * mask`.

Each identity uses the unchanged POINTWISE bar: normalized RMS and
maximum/NEMO-RMS both `<=1e-15`. Exact bit counts are diagnostic. A
source-ordered reconstruction that misses any bar invalidates the instrument;
it does not assign physics ownership.

The production comparison is released only from a SHA-pinned checkout that
contains the round-25-authorized literal coefficient builder and whose replay
puts the Coriolis input at bar. Bottom stress is `AT_BAR` only if the production
face coefficient, live depth, reciprocal, and applied increment all meet both
POINTWISE bounds. The final update is `AT_BAR` only if all its operands and
output do. A first DEBT operand is the ordered owner; a production replay not
yet available is `HELD`, never inferred from the oracle self-identity.

Only an AT-BAR bottom stress and final update release row 1.4. Only an AT-BAR
row 1.4 releases rows 2--6, then the free-surface-filter, momentum-RHS, and
tracer-tail chains in NEMO execution order. The identity, missing-stream, and
four-step-`nextafter` controls must be red-capable.

## Loader correction after the first score stopped

Blocks 1--3 completed with the registered 211/223 bracket, and all twelve new
streams are the required 90,944-byte full-halo products. The first scorer
stopped before emitting an artifact while loading the pre-existing
`spg_dump_zu_frc.bin`: 10,348 doubles cannot be reshaped to `(203,56)`.

This is a loader defect, not a writer defect. NEMO declares `zu_frc,zv_frc`
as `REAL(wp), DIMENSION(A2D(0))` at `dynspg_ts.F90:168`; `A2D(0)` is the
halo-free `(199,52)` domain. The new `zu_trd/zu_spg`, depths, coefficients, and
outputs are declared `DIMENSION(jpi,jpj)` at `:166-174` and remain full-halo.
The amended scorer uses the deterministic writer's interior loader for the two
frozen forcing streams only, asserts their exact 82,784-byte size, and records
both layouts in its receipt. No arm, bar, population, arithmetic, or
disposition rule changes.

The exact bracket SHA-256
`f9ff8f957c3228a882339b7ce5e5e423ad22f4e0d0e7f9839dc761f242841854`
is retained; rebuilding or rerunning NEMO would add no information. The
cleanliness gate is narrowed to tracked paths: staged or unstaged tracked edits
still stop scoring, while unrelated untracked campaign outputs do not. The
SHA-pinned producer/model-diff gate remains unchanged.
