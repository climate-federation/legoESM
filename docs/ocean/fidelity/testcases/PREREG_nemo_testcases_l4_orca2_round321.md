# PREREG — ORCA2 round 321: L3 (eddy-induced velocity) under Decision 122

Frozen before any ladder runs (Claude round; codex review: operator).
Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round321_l3_d122_gate.py`
(round-285 ladder, round-318 later-first classifier = DECISION 139's
safeguard, round-278 decision). CPU, fp64, libm, production JIT.

Import: lane B gridladder-vmix `b3e5011190..fb02c4b29f` package hunks
(L3 = `698ee4d7d8..fb02c4b29f`, after lane B's G9 merge), default-off,
commit `fcf91db15`. Measured before arms: the normalized step programs of
rung 0, OMT-0..4, rung 10, GYRE, VORTEX (flux, vector) and DINO are
identical to base `54a7595e0` (11/11), and rung 10's three production
steps are bit-identical (72 leaf-steps).

Which records run L3: rung 10 only (its ocean.output: ln_ldfeiv = T,
nn_aei_ijk_t = 21, rn_Ue = 0.03, rn_Le = 200 km; EXP00 namelist_cfg
lines 317-323). Rung 0 and OMT-1..4 record ln_ldfeiv = F.

Observed BEFORE this preregistration (trace only, not predictions;
`orca2_rounds/round321/probes/trace_*.log`):
- L3 traces on rung 10 (L2 + EIV, G9's stage-3 Krhs order off as shipped).
- L3 with G9's stage-3 Krhs order refuses on rung 10 (that order needs
  FCT's stage-3 written content order, which rung 10 keeps off).
- L3 refuses on rung 0 (same refusal as L2 in round 320); rung 0's record
  does not run EIV anyway.

So rung 10 runs L3 on the route where L2's isoneutral tendency enters as a
stage source rate, not lane B's G9 Krhs tuple; the bolus itself takes the
certified route (stage-3 pFu/pFv, then wzv).

| ID | Prediction | Falsifier |
|---|---|---|
| R321-P1 | L3 completes kt 1..10 on rung 10, both labels, finite. | refusal or non-finite state. |
| R321-P2 | L3 passes D122 (c)+(d) on rung 10, both labels (DECISION 139's later-first safeguard allowed); the other five pairs are unchanged cards. | an exact row lost, or first over-bar row away. |
| R321-P3 | L3 lowers rung 10's kt-10 stage-3 RMS T error (independent label). PLAUSIBLE: NEMO runs the bolus advection, the card does not. | RMS T error not lower. |
| R321-P4 | L3 changes rung 10's kt-10 max sea-level error by less than 20 % (a tracer transport). | a change of 20 % or more. |
| R321-P5 | Rung 10, N = 1, L3 selected: 0 non-finite entries in the T-loss gradient w.r.t. initial T, rn_Ud and aei0; reverse = forward (relative <= 1e-8). | any non-finite entry, or reverse != forward. |

Census (toward/away/equal) and every other number are registered, not
predicted. A pass that leans strongly away (census away > 2x toward) is
asked, not landed.
