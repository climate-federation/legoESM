# Preregistration — grid ladder G5, round 2 (gridladder_r32)

**Frozen:** 2026-10-15, before admission internals, forcing identity and ladders
are measured.  Record: `gridladder_rounds/round31/oracle_g5_r31` (operator run
of the corrected acquisition; `G5_ACQUISITION_PASS steps=1..10`).

Disclosure: the passivity check (step 1 of the round brief) was run BEFORE this
file was frozen, because the brief makes it the gate for everything else.
Result, measured: writer-free reference vs record twin A restart at step 10,
38/38 restart variables byte-identical (`check_records.restarts_identical`);
reference vs twin B also identical at steps 2, 4, 6, 8, 10.

## Predictions

1. Admission: every per-step frame (entry, surface, bt, stage 1-3) is
   byte-identical across twins A/B/C for kt 1..10; NEMO's stage-3 frame of kt
   equals its entry frame of kt+1 (kt 1..9); every restart equals the stage-3
   and barotropic frames of its step.
2. Forcing identity: every ocean-read field the card takes from the record
   (utau, vtau, taum, qsr, qns, emp, rnf, sfx, fr_i, rnf_tsc) is 0 unequal at
   every kt; the U/V stress statement on the card's input reproduces NEMO's
   recorded utauU/vtauV with 0 unequal; `rCdU_ice` is finite and nonzero on
   ice-covered wet cells at every kt and has no card input.
3. First non-bit statement (round 1's frozen prediction, unchanged): the first
   external-mode use of the implicit ice-ocean top drag `rCdU_top`
   (`dynspg_ts.f90:1431-1432`), visible at kt 1 as ssh / barotropic-velocity
   mismatches confined to ice-covered columns, while open-ocean columns stay
   bit-identical.
4. G5 is NOT certified this round.

## Falsifiers

Any twin difference or passivity failure (record unusable); any forcing-field
mismatch; kt 1 bit-identical everywhere; a first kt-1 mismatch in a column with
no ice; `rCdU_ice` zero on all ice-covered wet cells.
