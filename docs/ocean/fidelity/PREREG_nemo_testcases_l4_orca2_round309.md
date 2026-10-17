# PREREG — ORCA2 round 309: OMT-0 under round 308's set; rung V3 under Decision 122

Frozen before any measurement (Claude round; codex review: paused (quota)).
Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round309_omt0_v3_d122_gate.py`.
CPU, fp64, libm, production JIT.  Records as round 308; OMT-0 = its admitted
round-203 kt 1..10 frame record (`orca2_omt0_frames_10step_a_np2`).

Resolved before measuring (read off the configs and the decks):
- OMT-0 runs NEMO's linear momentum program (ln_dynadv_OFF -> np_LIN_dyn,
  flux form, planetary EEN); the card carries round 308's eight selections.
- Double diffusion: only the rung-10 deck sets ln_zdfddm = .true.; every
  card has it off today (rung 10 lists it unmeasured).
- EVD: every deck and card runs it (rn_evd 100, nn_evdm 0).

| ID | Prediction | Falsifier |
|---|---|---|
| R309-P1 | OMT-0, round-308 set vs without: 0 exact rows lost, first over-bar row unchanged (D122 (c)+(d) pass). | any exact row lost, or first over-bar `away`. |
| R309-P2 | OMT-0 NOADV (set minus vector Krhs order and ZAD association) equals the full set row for row (NEMO's np_LIN_dyn never calls dyn_adv). | any row differs. |
| R309-P3 | EVDSE refuses on rung 0 and OMT-1..4 (constant-mixing cards build no step-entry bn2 bundle). | it runs. |
| R309-P4 | EVDSE runs on rung 10, both labels, and passes (c)+(d). | refusal, exact loss, or first over-bar away. |
| R309-P5 | DDM and V3 refuse on rung 10 (it runs internal-wave mixing; the imported literal DDM refuses with IWM on). | either runs. |
| R309-P6 | The default-off import leaves GYRE (ten-step residuals, day 30), DINO and VORTEX step jaxprs identical. | any difference. |
