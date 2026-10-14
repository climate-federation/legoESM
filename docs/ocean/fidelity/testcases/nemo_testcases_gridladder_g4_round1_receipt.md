# Grid ladder rung G4 — round 1 receipt (gridladder_r17)

Status: **STOPPED_FOR_RECORD.** No NEMO record of G4 exists; nothing was
measured against NEMO.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g4_round1.md
(fd529aead, corrected d19a1ce7f after review; predictions unchanged).
Acquisition: scripts/validate/ocean_fidelity/testcases/nemo_testcase_g4_orca2sbc/run.sh
(preflight PREFLIGHT_OK at d19a1ce7f); evidence will land in
gridladder_rounds/round17/oracle_g4_r17.

## What was built

| item | content |
|---|---|
| deck | G3's, plus ORCA2's &namsbc switches with nn_ice = 0 and ORCA2's &namsbc_blk, &namtra_qsr, &namsbc_ssr, &namsbc_rnf verbatim; nothing else moved |
| binaries | G1's round-4 pair reused (keys = ORCA2's minus key_si3); no build |
| inputs | 14 files pinned (CORE-II normal year x7, two weight files, chlorophyll, sss_data, runoff_core_monthly, the T/S climatology) + domain + namelist_ref |
| runs | smoke 2 steps; reference 100-step deck (NEMO's stop sets the record length, cap 10, even); twins a/b/c at that length |
| card | ORCA2G4-sbc = G3 card + ORCA2's RGB light + forcing labels; not execution-ready (forcing family unmeasured) |

## Searched before building

Existing NCAR bulk (`bulk_flux_omip.py`), NEMO fldread (`forcing/nemo_fld_read.py`),
RGB light config, the ORCA2 card's forcing labels and G3's acquisition: all
reused or named; no new forcing implementation written.

## Tests (synchronous)

Focused battery (G4, G3, G2, G2 ladder, G1 card, recipe unit): 130 passed,
1 failed — `test_standin_card_builds_and_states_every_selector`, which also
fails at the base commit (it expects the pre-G3 check name `uniform_T`).
Citation gate: 17 passed (no pinned recipe line moved). Non-vacuity: every
planted forcing line is refused by the card check; run.sh's three deck
comparisons each refuse a planted line; the 12 reader tests fail with the
reader fix removed.

## Review

Single review (codex): **"VERDICT: findings"** (codex_review_r17.log) — six
findings, all confirmed and fixed in d19a1ce7f: namelist_ref unpinned
(HIGH), finished-reference error check, frame hashes missing, reader
weakened, two prereg citations.
