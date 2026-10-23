# ORCA2 round 329 — the rung-10 comparison builds the record's own initial sea ice (DECISION 157)

Date: 2026-10-23

Base: `735cab5ba22e`; code `02acc8a1b` (+ re-anchor `a48dd2956`, arms there;
gradient probe `330128d84`).  Preregistration:
`PREREG_nemo_testcases_l4_orca2_round329.md` (frozen before the arms; the
kt-1 entry check E1 was run once while wiring, disclosed there).

Disposition: **LANDED-pending** (operator review + push gate + DINO month gate).

## Answer

**CONFIRMED.**  The independent label's kt-1 entry offset is closed: the
comparison card's entry SSH is bit-identical to NEMO's on all 26,640 cells
(16,433 wet, every one at -0.10891439254081828 m).  The shipped card keeps
-0.09343505872684969 m, unchanged.

## Change

- The card's inline initial-ice/SSH block is now one recipe function; the
  shipped card calls it with its unchanged five-category ponds-on values
  (`nemo_testcase_recipe.py:1834-1839`).
- The comparison card (rung 10 only) calls that same function with the
  record's &namini values, parsed from the record's `ocean.output`: jpl = 1,
  ln_pnd = F, rn_thres_sst 2, rn_ati_ini 0.997/0.8, rn_hti_ini 2.5/1.0,
  rn_hts_ini 0.2/0.1; pond area = rn_apd_ini * a_i
  (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:324`),
  zero because ln_pnd = F
  (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:343-346`);
  SSH lowered by the global mass as NEMO does
  (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:442-459`).
- Plants: a missing `ocean.output`, a missing key, a duplicated key, and a
  file-based initial ice each refuse (4/4 tests).  Removing the selection makes
  the entry test fail (shown, then restored).

## Shipped cards unchanged (S1, CONFIRMED; `orca2_rounds/round329/jaxpr/`)

- Normalized one-step programs base vs round: 7/7 identical (rung 0, OMT-0..4,
  rung 10).  Only the ORCA2 card builder calls the changed code; GYRE, VORTEX
  and DINO do not.
- Rung 10, 3 production steps: 72/72 leaf-steps bit-identical (SHA-256).

## Decision 122, seven pairs (S1/D1; `orca2_rounds/round329/arms/`)

Control = round 326's comparison card (drag, L3, nn_mxlice = 2, card ice);
candidate = the same plus the record's initial ice.  `decide7_recordice.json`:
`PASS_R278_D122_DECISION`, SHA-256 `7d2154ac1bfc163c…`.

| pair | exact rows lost | first over-bar (D139 recomputed) | census toward/away/equal |
|---|---:|---|---|
| rung 0, OMT-1..4 | 0 | equal (cards unchanged) | 0/0/0 |
| rung 10 given entry | 0 | equal, kt 1 stage-1 v | 0/0/0 (arms bit-identical rows) |
| rung 10 independent | 0 | moved later: kt 1 entry SSH -> kt 1 stage-1 v, toward | 196/0/0 (early 56/0, late 140/0) |

## kt-10 full-domain entry errors, rung 10 (K1, CONFIRMED; `kt10_rung10.json`)

| label, arm | SSH max (m) | T max (K) | S max | u max (m/s) | v max (m/s) |
|---|---:|---:|---:|---:|---:|
| independent, before | 1.853309e-02 | 1.142969e-02 | 1.879389e-03 | 5.295100e-03 | 3.147371e-03 |
| independent, after | 3.734978e-05 | 7.703007e-05 | 1.223755e-05 | 3.046847e-03 | 3.149083e-03 |
| given entry, before = after | 3.734978e-05 | 7.703007e-05 | 1.223755e-05 | 3.046847e-03 | 3.149083e-03 |

RMS, independent before -> after: SSH 1.216e-02 -> 1.104e-06 m, T 2.108e-05 ->
3.465e-07 K, S 4.227e-06 -> 6.621e-08, u 1.421e-05 -> 4.908e-06, v 1.184e-05
-> 7.115e-06 m/s.  After the change, all 200 ladder rows of the two labels
carry identical metrics (max, RMS, unequal count, first unequal index).
The independent v maximum rises by 1.7e-6 m/s to the given label's value.

## Gradients (G1, CONFIRMED; `orca2_rounds/round329/grad/`)

Rung-10 N = 1 on the comparison card from its own initial state (record ice,
SSH -0.10891439254081828 m): 0 non-finite (state cotangents and leaves, both
losses); reverse = forward 4.0e-14 (T direction), 1.4e-15 (SSH direction),
1.7e-15 (rn_Ud, -4.5117e-05); forward parity 0 non-bit leaves; peak 74.3 GiB.
SHA-256 `44b388ff81617cda…`.  Every other card is unchanged (S1).

## Gates

- Receipt citation gate on this receipt: see the machine line below.
- Default battery (landing set): 31 + 96 + 20 = 147 passed (the third part
  printed `20 passed in 565.08s`; its 598 s wrapper timeout fired during
  interpreter exit).  New and earlier comparison-card tests: 26 passed.
- Private-import ratchet: 6 passed.
- Eleven citation-map entries re-anchored for the +43-line shift (rigid), and
  the round-327 initial-ice citation now points at the builder's call.

codex review: operator

## CHOICES

- Read the record's `ocean.output` (resolved echo) rather than its namelist — ASKED (prompt: "its namelist / ocean.output").
- Record path is a module constant (the round-5 record the hm_i admission already uses) — UNASKED.
- Cards other than ORCA2 keep their SSH under the comparison card (GYRE stands in for earlier tests) — UNASKED.
- Ponds-on records would feed rn_apd_ini * a_i; non-zero pond lids refuse — UNASKED (not exercised).
- jpl is read and recorded but the card's mass sum does not depend on it — UNASKED.
- Refuse unless ln_iceini = T and nn_iceini_file = 0 — UNASKED (extends the asked refusal).
No configuration value of any shipped card moved.

## OPEN

- Rung 10's first over-bar row is now kt 1 stage-1 v on both labels; the
  dyn_zdf walk waits for round 328's acquisition.

UNVERIFIED: identical row metrics between labels are not an array diff.
