# PREREG — vertical-mixing lane V2 (preregistered round 12, vmix_r12)

Frozen before any V2 measurement. No V2 record exists yet; nothing is scored this round.
Decisions 128-130 in force. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round12/`.

**V2 = certified V1 + ORCA2's enhanced vertical diffusion, exactly as ORCA2 sets it**
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:390-392`: `ln_zdfevd = .true.`, `nn_evdm = 0`, `rn_evd = 100.`).
Nothing else moves. It is the first of ORCA2's vertical-mixing switches V1 does not run, in round 1's order
(EVD, DDM, IWM, `nn_havtb`). NEMO: `zdf_evd` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:56`), called at
`zdfphy.f90:359` in `zdf_phy` (:296) after the closure copy (:348-351).

Card: `build_orca2v2_evd_card` = V1's card with the ORCA2 identity's own EVD arm kept (trigger
`MIN(rn2,rn2b) <= -1e-12`, avt replaced by `rn_evd`, `zdfevd.f90:107-110`) and `nu_conv = 0` for
`nn_evdm = 0` (`zdfevd.f90:121`: avm untouched). Record: `gridladder_vmix_rounds/round12/oracle_v2_r12`
(acquisition `scripts/validate/ocean_fidelity/testcases/nemo_testcase_v2_orca2evd/run.sh`).

- **P1 (record).** The V2 run resolves `ln_zdfevd = T`, `nn_evdm = 0`, prints `zdf_evd`, keeps every V1
  switch, completes >= 10 steps, and its three twins agree on every frame. Falsifier: the script refuses.
- **P2 (EVD is not inert).** At some kt in 1..10, NEMO's V2 T or S differs from NEMO's V1 record in at least
  one wet cell. Falsifier: V2 and V1 records bit-identical kt 1..10 (then this window cannot certify EVD;
  report it and ask for a longer window, do not certify).
- **P3 (first owner, PLAUSIBLE).** If the V2 card misses NEMO at any kt 1..10, the first non-bit statement is
  EVD's trigger N2: the card's EVD builds its own `bn2`, not the per-operation-rounded `rab_3d_t`/`bn2_t` V1
  selected for TKE (that flag is read only by TKE's N2). Falsifier: the first non-bit row is owned elsewhere
  (named by the walk), or V2 is bit-for-bit kt 1..10 under both labels (then V2 certifies as built).
- **P4.** The V2 card leaves every other card unchanged: full ORCA2 and DINO flag-off one-step jaxprs
  identical to `e01d443d3008`; V1's card fields and initial state unchanged (V1 card test).
