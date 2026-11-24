# PREREG — ORCA2 round 320: L2 (isoneutral) and L4 (geothermal) under Decision 122

Frozen before any ladder runs (Claude round; codex review: operator).
Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round320_l2_l4_d122_gate.py`
(round-285 ladder, round-318 later-first classifier = DECISION 139's
safeguard, round-278 decision).  CPU, fp64, libm, production JIT.

Import: lane B `3cfd91d074..51b2582f5e` (L2) and lane A
`7a31b67ef..b4f70973d0` (L4) package hunks, default-off, commits
`fc4d58556` and `8a6fbf66e` (two order-preserving lines).  Measured before
arms: the step jaxprs of rung 0, OMT-0..4, rung 10, GYRE, VORTEX (flux,
vector) and DINO are identical to base `246f73429` (11/11).

Which decks set each piece (resolved namelist_cfg of each card's record):
isoneutral on rung 0 and rung 10 (OMT-1..4 state ln_traldf_off);
geothermal on rung 10 only.

Observed BEFORE this preregistration (trace-only feasibility, not
predictions; `orca2_rounds/round320/probes/trace_*.log`):
- ISO traces on rung 10.
- ISO refuses on rung 0: L2's Kmm Redi geometry needs the stored stage-3
  tracer thickness, which rung 0 keeps off (no forcing module).
- GEO and BOTH refuse on rung 10: lane A's guard refuses geothermal with any
  tracer lateral diffusion (NEMO adds tra_bbc after tra_ldf; this model adds
  the diffusion increment afterwards).

So the only ladder arm that can run is ISO on rung 10 (both labels).  OMT-1..4
and rung 0 carry no piece: their step programs are the base programs (above).

| ID | Prediction | Falsifier |
|---|---|---|
| R320-P1 | ISO completes kt 1..10 on rung 10, both labels, finite. | refusal or non-finite state. |
| R320-P2 | ISO passes D122 (c)+(d) on rung 10, both labels (DECISION 139's later-first safeguard allowed). | an exact row lost, or first over-bar row away. |
| R320-P3 | ISO lowers rung 10's kt-10 stage-3 RMS T error (independent label). PLAUSIBLE: the shipped isoneutral block differs from NEMO's in coefficient shape, MSC and slopes (round 314). | RMS T error not lower. |
| R320-P4 | ISO changes rung 10's kt-10 max sea-level error by less than 20 % (a tracer operator). | a change of 20 % or more. |

Census (toward/away/equal) and every other number are registered, not
predicted.  A pass that leans strongly away (census away > 2x toward) is
asked, not landed.
