# Grid ladder rung G4 — round 4 receipt (gridladder_r20)

Status: **STOPPED_FOR_DECISION.** The per-step flux record is admitted
(no NEMO re-run). Nothing landed in the model; forcing NOT wired into the
card (Decision 123 pending). G4 is NOT certified.
Record: gridladder_rounds/round19/oracle_g4_r19. Evidence: gridladder_rounds/round20/.
Decisions 120, 121, 123 pending; nothing here depends on their answers.
Diagnosis-only round (brief): no preregistered landing, so no new PREREG.

## Take-aways

1. **The refusal was the check's.** Every moved word in the substep streams
   is halo storage NEMO never initialises (values about 1e-310). No
   inner-domain word, no stage array, restart or diagnostic moved.
   Re-admitted: ACQUISITION_DONE.
2. **Shortwave is NEMO's bit for bit at all 10 steps** (16,433 wet cells).
3. **Wind stress is over the bar at all 10 steps**, on every wet cell, in
   two populations: calm, near-neutral air (up to 87 % relative) and cold
   air over warm sea (up to 2.4e-3).
4. **Two NEMO statements the shared bulk lacks carry it:** NEMO has no
   wind floor in its neutral drag (the shared kernel floors at 0.5 m/s),
   and NEMO floors air humidity at 1e-6 (it binds where the mapped humidity
   is negative). Three last-bit forms come next.
5. **No ladder, no certification**: the card still has no forcing; Decision
   123 chooses how it gets one.

## 1. Admission of the round-19 record

The operator's refusal: whole-file byte equality of twin a's kt = 1 solver
substep stream against round 17. Mapped word by word with the stream reader:

| twin, kt | bytes moved | 8-byte file chunks | field words moved | inner-domain words | band next to inner | outer halo ring |
|---|---|---|---|---|---|---|
| a, 1 | 938 | 335 | 268 | 0 | 0 | 268 |
| a, 2 | 10,694 | 3,845 | 2,998 | 0 | 0 | 2,998 |
| b, 1 | 6,295 | 1,066 | 926 | 0 | 329 | 597 |
| b, 2 | 4,842,691 | 737,201 | 647,393 | 0 | 290,639 | 356,754 |
| c, 1 | 1,206 | 335 | 268 | 0 | 0 | 268 |
| c, 2 | 13,692 | 3,845 | 2,998 | 0 | 0 | 2,998 |

- The operator's counts reproduce exactly (938 / 335 and 10,694 / 3,845);
  their "words" are file-aligned 8-byte chunks, which straddle field words
  (76-byte header), hence 335 chunks for 268 words. Every moved byte lies
  inside a moved field word.
- Twins a, c: cells (jj, ji) = (1, 1..4) only, the outermost halo row, in
  3 (kt 1) and 15 (kt 2) solver work arrays (`zCdU_u/v`, `htp2_e`,
  `hup2_e`, `zhU`, `sshu_a`, `cor_u`, `trd_u`, `zu_spg` and V twins).
- Twin b also moved words in the band next to the inner domain, in 11 work
  arrays (zCdU_u/v, htp2_e, sshu_a/sshv_a, zu_spg/zv_spg, cor_u/v,
  trd_u/v). None reaches a NEMO result: every stage array of every step
  (halos included), every restart, and all 47 variables of NEMO's
  diagnostic files are byte-identical in all three twins. The solver
  stencil bands that read zhU/zhV moved 0 words (the reviewer measured this).
- Fix (check, not NEMO): the substep streams are compared on the consumed
  support, the twin-admission standard (inner domain of full planes, other
  arrays whole), with equal headers. Diagnostics are now compared too. Halo
  and band counts go to `g4_round19_passivity.json`. A one-word inner plant
  must refuse, and it does.
- `run_r19.sh --admit-existing`, NO NEMO run: G4R19_PASSIVE, 654,851 halo
  words counted, 9 of 9 record plants fired, ACQUISITION_DONE.

## 2. Forcing identity at every step (INDEPENDENT label)

Shared code: NEMO's fld_read mapping and NCAR bulk (the ORCA2 lane's
certified-path operators) on the CORE records NEMO's log names at each
bulk step (6-hourly record kt 1→1, 3→2, 5→3, 7→4, 9→5; daily record 1,
then 2 at kt 9). Ocean surface state: NEMO's own surface mean from the
record (round-18 convention; the card cannot step forced). The bulk runs at
kt = 1, 3, 5, 7, 9 (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcblk.f90:554`);
even steps carry the previous bulk step's fluxes. Bar: 1e-15 normalised.

Unequal wet cells / normalised max (T fields over 16,433 wet cells; U/V
fields over every face NEMO's averaging can make nonzero):

| kt | qsr | utau (T) | vtau (T) | utauU (read) | vtauV (read) |
|---|---|---|---|---|---|
| 1, 2 | 0 / 0 | 16,433 / 5.1e-4 | 16,433 / 1.0e-3 | 16,869 / 2.6e-4 | 16,849 / 5.1e-4 |
| 3, 4 | 0 / 0 | 16,433 / 3.4e-4 | 16,433 / 7.0e-4 | 16,869 / 3.6e-4 | 16,849 / 7.0e-4 |
| 5, 6 | 0 / 0 | 16,433 / 3.4e-4 | 16,433 / 7.1e-4 | 16,869 / 2.0e-4 | 16,849 / 3.7e-4 |
| 7, 8 | 0 / 0 | 16,433 / 4.0e-4 | 16,433 / 5.7e-4 | 16,869 / 2.1e-4 | 16,849 / 3.7e-4 |
| 9, 10 | 0 / 0 | 16,433 / 4.7e-4 | 16,433 / 5.3e-4 | 16,869 / 2.4e-4 | 16,849 / 3.5e-4 |

The ocean reads the U/V stress (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:198-199`).
The card has no T-to-U stress operator. The instrument mirrors
`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcmod.f90:557-560` on stress masked to
the ocean as NEMO masks it (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcblk.f90:796`).
That mirror reproduces NEMO's own U/V stress from NEMO's own T stress bit
for bit at every step.

Stress population split, by SST minus air temperature (max relative error,
wet cells per bin; vtau's maxima are the same):

| bulk kt | below 5 K | 5-20 K | above 20 K |
|---|---|---|---|
| 1 | 14,451 cells, 0.70 | 815, 6.8e-4 | 1,167, 2.4e-3 |
| 3 | 14,495, 0.68 | 801, 6.5e-4 | 1,137, 2.3e-3 |
| 5 | 14,568, 0.87 | 762, 5.2e-4 | 1,103, 2.4e-3 |
| 7 | 14,577, 0.74 | 787, 5.4e-4 | 1,069, 2.4e-3 |
| 9 | 14,550, 0.79 | 852, 9.7e-4 | 1,031, 2.4e-3 |

Not scored, no card-side operator (Decision 123's subject), NEMO's wet
range at kt 1: taum 2.3e-5..2.59 Pa and wndm 0.12..30.5 m/s (read only by
TKE, off in this deck); qns -1489..119 W/m2; emp -1.4e-4..2.0e-4;
rnf 0..5.7e-4; erp ±4.63e-5; qrp ±5.3 W/m2; runoff heat 0..1.5e-5, runoff
salt 0. Without sea ice, sfx and fr_i are 0 on every wet cell at every step,
as the card's would be. The barotropic composites need all of the above.

## 3. The NCAR stress operand (GIVEN-NEMO-ENTRY walk)

Every statement of `turb_ncar` was replayed on NEMO's own recorded
operands (entry frame, previous iteration), bulk kt 1, 3, 5, 7, 9, five
iterations each. Rows with unequal wet cells:

| statement | what differs | cells (kt 1/3/5/7/9) | max relative |
|---|---|---|---|
| `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90:141` entry neutral Cd | sixth power `:260-261` | 8 / 2 / 4 / 1 / 3 | 2.5e-16 |
| `:148` entry Ce, `:225` each iteration's CeN | multiplication order `:305` | about 8,600 | 3.9e-16 |
| `:159` humidity floor 1e-6 | absent in shared code | 182 / 145 / 163 / 164 / 153 | mapped q negative on all |
| `:195` psi_m | `:310-345` form | about 3,300 | 6.6e-12 |
| `:207` neutral wind | log/exp form, `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbc_phy.f90:1198-1213` | about 8,300 | 1.1e-15 |
| `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90:209` in-loop neutral Cd | no wind floor in NEMO, 0.5 m/s in shared | up to 59 / 51 / 34 / 37 / 36 | 0.49 |

Bit-identical on NEMO's operands: wind floor, virtual-temperature
difference, both stability flags, entry Ch, 1/L, zeta clamp, root CdN, Cd,
root Cd, psi_h term, Cd ratio, ChN, and both final updates of Ch and Ce.

- **First non-bit operand, program order:** the entry neutral drag
  coefficient, `:141` (function `:241-274`). CONFIRMED: NEMO's literal
  form `zw6 = (zw*zw*zw)**2` (`:260-261`) reproduces NEMO on every wet cell;
  the shared kernel's U**6 does not (8 cells at kt 1).
- **Carries the large stress error:** `:209`. CONFIRMED: every cell with a
  large error has NEMO's neutral wind below 0.5 m/s (48 of 48 at kt 1,
  iteration A; 30 of 30 at kt 9, iteration E; range 0.25-0.46 m/s). NEMO's
  `cd_n10_ncar` has no floor; NEMO floors only the neutral wind, at
  0.25 (`:207`). The shared kernel floors its input at 0.5, so its drag is
  up to 49 % low there.
- **Carries the cold-air error:** `:159`. CONFIRMED in place: the floor
  binds where NEMO's own mapped air humidity is NEGATIVE (all 182 cells at
  kt 1, down to -6.1e-4 kg/kg; 153 at kt 9, down to -8.8e-4), all with SST
  minus air between 17 and 39 K. The shared iteration uses q as given.
- Multiplication order (`:305`) and log/exp neutral wind (`:207`):
  CONFIRMED the same way (NEMO's literal form 0 unequal cells, shared form
  about 8,600 and 8,300). psi_m (`:310-345`, nested square roots vs a
  quarter power): PLAUSIBLE, not replayed in NEMO's form.
- Whole shared chain on NEMO's entry vs NEMO's final Cd: 3,776-3,879
  unequal cells; max 0.68-0.87 below 5 K, 3.7e-4..6.6e-4 at 5-20 K,
  3.9e-4..6.5e-4 above 20 K. The stage after the iteration (air density,
  stress assembly) was not walked.

Nothing landed: the bulk change is Decision 123's option 2.

## 4. Tests (synchronous)

Focused battery (G1 tools, G2, G2 ladder, G3, G4), JAX on CPU: 84 passed,
9 deselected (80 before + 4 new). Non-vacuity: re-inverting the inner-domain
cut and shifting one citation by a line each turn a test red. The
instrument's two plants fire (exit 1). The program-check test refuses a
wrong cadence, wrong height and a missing iteration (exit 73).

## 5. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r20.log).
Each checked before acting:
- HIGH, the consumed support is narrower than the solver's stencil:
  PARTLY CONFIRMED. Twin b moved words in the next-to-inner band of 11
  work arrays (table above). The reviewer measured 0 in the bands the
  solver reads for zhU/zhV. Every downstream result is byte-identical, and
  that includes the diagnostics now checked. Fixed: equal headers, a
  diagnostics comparison and band counts in the JSON.
- MEDIUM, U/V faces dry here and wet next door were not scored: CONFIRMED
  (578 U, 558 V). Fixed: scored on every face NEMO can make nonzero, with
  a test for that coastline orientation.
- MEDIUM, hard-coded cadence, halo width, heights, wave drag, iteration
  count; no CPU refusal in the walk: CONFIRMED. Fixed: read from the run
  (log, decks, recorded frames), refusals exit 73.

## 6. Choices

ASKED (round brief): passivity fixed in the check on the consumed support;
re-admit without NEMO; diagnosis only.
UNASKED, each offered for revert: (1) the passivity check now also compares
NEMO's diagnostic files (a new refusal condition); (2) the U/V stress
mirror of the sbcmod statement exists only in the instrument; (3) the
SST-minus-air bins 5 K and 20 K.

No gate exists for: dual review (single, lane rule 6), claim review before
code, non-vacuity (shown above).

UNVERIFIED:
- psi_m's last-bit cause (not replayed in NEMO's form);
- the stage after `turb_ncar` (air density with the floored humidity,
  taum/wndm factorisation) — not walked.

## OPEN

Decision 123. Under (1), feed the record's per-step surface fields to the
card and ladder kt = 1..10. Under (2), land, one at a time, the two NEMO
statements in section 3 (no neutral-drag wind floor, humidity floor), then
the last-bit forms.

## 7. Citation gate and evidence hashes

Citation gate on this receipt (from section 1): **PASS**, 16 citations, 0
failures, 0 unmapped, map audit clean; planted shift of `:209` exits 1;
gate tests 17 passed.
Instrument runs at commit 642e8c619 (round20/legoesm_commit); admission
re-run at the same commit.

SHA-256 prefixes, round20/: admit_existing.log=5eb04c55071a44c1, forcing_identity.json=0faae13d7ddc7500, forcing_walk.json=a9c9acedecfde45d, plant_identity.log=d1c3f86e9c0f34eb, plant_walk.log=d1c3f86e9c0f34eb, codex_review_r20.log=8ee89a96bf0a7fce, tests_focused.log=c7ee4c796f15ca20, citation_receipt.json=eb0c96dc0559fdb1, citation_plant.log=498a255625e8a674, tests_citation_gate.log=1b546ad510fdce01, round19/oracle_g4_r19/g4_round19_passivity.json=75337ee077c8f3f8.
