# PREREG — ORCA2 round 323: L8 stage-3 Krhs route, then L3 and L5, under Decision 122

Frozen before any ladder arm runs (Claude round; codex review: operator).
Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round323_l8_l3_l5_d122_gate.py`
(round-285 ladder, round-318 later-first classifier = DECISION 139's
safeguard, round-278 decision). CPU, fp64, libm, production JIT.

Imports, default-off:
- `a4fa89432`: lane B L1 `40b5e44139`, `53624609d0`, `3cfd91d074` (BBL
  trend in stage-3 Krhs, pivot ghost) and lane A L4b `f0549b581d`,
  `ff080c094c`, `288311ef9b`, L6 `11c7147545`, `945f885f94`, L8
  `ce9636bc1f` package hunks; recipe: only the shared climatology-at-step
  and SHA-pinned damping builder. Normalized step programs of rung 0,
  OMT-0..4, rung 10, GYRE, VORTEX (flux, vector), DINO: 11/11 identical to
  base `ea02b6e6f`.
- this commit: lane B L5 `acda81beb`, `bedbcf650`, `7b09b4654`,
  `43071fbf2`, `1075ccf0f` package hunks (recipe: one public constructor
  with lane B's L5 card arguments) and the public L8 selection.
  Step programs and rung-10 forward bits: in the receipt.
- L7 (river-mouth mixing) not imported: inert in ORCA2 (socoefr0 all zero).

Which records run each piece: rung 10 only. Its ocean.output states
ln_trabbl = T, ln_trabbc = T (nn_geoflx = 2), ln_tradmp = T (nn_zdmp = 0),
ln_ldfeiv = T, ln_mle = T; rung 0 and OMT-1..4 state all five F.
Inputs: rung 10's deck links ORCA2_ICE_v5.0.0's geothermal_heating.nc
(sha256 e369762f...6fffb) and resto.nc (sha256 f4b6aec4...8b634, the pin
lane A's builder checks); the ladder refuses either on a SHA mismatch.

Observed BEFORE this preregistration (construction only, not predictions):
- rung 10 constructs with L8, and with L8 + L5;
- rung 10 REFUSES L8 + L3 at construction: lane A's L8 guard admits only
  the exact certified L2 isoneutral block (no eddy coefficient, no
  Treguier, centred bolus); L3's block carries all three. L8 + L3 is a
  composition no ladder rung certifies. It is not measured; it is asked.

Arms: control (shipped card, L2), L8, L8 + L5.

| ID | Prediction | Falsifier |
|---|---|---|
| R323-P1 | L8 completes kt 1..10 on rung 10, both labels, finite. | refusal or non-finite state. |
| R323-P2 | L8 passes D122 (c)+(d) on all seven pairs; the other five pairs are unchanged cards (0 rows moved). | an exact row lost, or first over-bar row away. |
| R323-P3 | L8 lowers rung 10's kt-10 stage-3 RMS T error (independent). PLAUSIBLE: NEMO runs geothermal and damping, the card does not. | RMS T not lower. |
| R323-P4 | L8 changes rung 10's kt-10 max sea-level error by less than 20 %. | 20 % or more. |
| R323-P5 | L8 + L5 completes, finite, and passes D122 (c)+(d) on all seven pairs against L8. | refusal, non-finite, exact row lost, or first over-bar row away. |
| R323-P6 | L5 (on L8) lowers rung 10's kt-10 RMS T error (independent). PLAUSIBLE only: ten steps of a mixed-layer restratification. | RMS T not lower. |
| R323-P7 | The final shipped rung-10 card, N = 1: 0 non-finite entries in the T-loss gradient w.r.t. initial T and rn_Ud; reverse = forward (relative <= 1e-8). | any non-finite entry, or reverse != forward. |

Census (toward/away/equal) and every other number are registered, not
predicted. A pass that leans strongly away (census away > 2x toward) is
asked, not landed.
