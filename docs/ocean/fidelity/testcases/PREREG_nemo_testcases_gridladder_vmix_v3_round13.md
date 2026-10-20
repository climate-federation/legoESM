# PREREG — vertical-mixing lane V3 (preregistered round 13, vmix_r13)

Frozen before any V3 code or measurement. No V3 record exists yet; nothing is scored this round.
Decisions 128-130 in force. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round13/`.

**V3 = certified V2 + ORCA2's double-diffusive mixing, exactly as ORCA2 sets it**
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:393-395`: `ln_zdfddm = .true.`, `rn_avts = 1.e-4`, `rn_hsbfr = 1.6`).
Nothing else moves. Second of the switches V1 does not run, in round 1's order (EVD, DDM, IWM, `nn_havtb`).
NEMO: `zdf_ddm` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:59`), called at `zdfphy.f90:362-363` in `zdf_phy`
(:296) after `zdf_evd` (:359) and before `zdf_iwm` (:372); with it off NEMO sets `avs = avt` (:365-367).

Survey facts (read off the code, not measured):
- `zdfddm.f90`, `zdfphy.f90`, `trazdf.f90`, `zdf_oce.f90` are byte-identical in ORCA2_OMIP_L4, V1's record build
  and G5's writer-free build; `zdfddm.o` is byte-identical in all three (flags `-fdefault-real-8 -O3
  -fno-tree-vectorize`, no `-march`: literals are double, no fused multiply-add).
- `zdf_ddm` writes a SEPARATE salt diffusivity `avs` (:170), adds to `avt` (:171) and to `avm` (:172).
  `trazdf` builds a second matrix for salt from `avs` when `ln_zdfddm` (`trazdf.f90:175`, :198-213).
  `avs` is recomputed every step from that step's T/S, alpha/beta and N2: it is NOT carried state.
- Its operands are `bn2`'s: the same 1-D `zrw` (`zdfddm.f90:118-120` = `eosbn2.f90:1632-1633`), `zaw`/`zbw`
  (`zdfddm.f90:122-125` = `eosbn2.f90:1635-1636` times `tmask(jk)*tmask(jk-1)`), on `rab_n = rab_b` and
  `rn2 = rn2b` at `Kmm = Nbb` (`stprk3.f90:170-174`, `:179`).
- Our repo's double-diffusion module is a documented hybrid (Large/CVMix cubic for fingering, sign-preserving
  ratio, no momentum term) and the NEMO-literal tracer solve raises on distinct heat/salt matrices.

The card change (V3-selected, every other card unchanged):
- a NEMO-literal `zdf_ddm` form in the EXISTING double-diffusion module (extend, not a second module), fed by
  the card's existing step-entry bn2 (V1's per-operation-rounded `rab_3d_t`/`bn2_t`): its ΔT/ΔS
  contributions and rn2; EXP through the precision policy's scalar libm (as V1's EXP);
- composed in the shared K-profile builder after EVD, before IWM (NEMO order); avs leaves through a new
  opt-in return; avm gets `MAX(zavft+zavdt, zavfs+zavds)`;
- the NEMO-literal tracer pair solve gains an optional second (salt) diffusivity; same matrix assembly.
No new model state (avs is step-local, like avt).

- **P1 (record).** The V3 run resolves `ln_zdfddm = T`, `rn_avts = 1e-4`, `rn_hsbfr = 1.6`, prints
  `use double diffusive mixing: avs /= avt` (`zdfphy.f90:257`), keeps every V2 switch, completes >= 10 steps,
  three twins agree on every frame. Falsifier: the acquisition refuses.
- **P2 (DDM is not inert).** (a) NEMO's kt-1 post-`zdf_phy` frame has avs != avt in at least one wet interior W
  cell; (b) at some kt in 2..10 NEMO's V3 T or S differs from NEMO's V2 record in at least one wet cell.
  Expected large: NEMO maps a stable single-signed interface (zdt/zds <= 0) to zrau = 1e-20, i.e. the
  diffusive-layering branch, adding 1.3635e-6 m²/s to avt there. Falsifier: either count is 0 (then report,
  do not certify).
- **P3 (first owner, PLAUSIBLE).** If V3 misses NEMO at kt 1..10, the first non-bit statement is one of
  `zdfddm.f90:159-172` — most likely the double EXP at :166 (scalar libm vs the card) or the `zrr*zrr*zrr`
  power order at :161-163. Falsifier: the walk names another owner, or V3 is bit-for-bit kt 1..10 under
  both labels (then V3 certifies as built).
- **P4.** Flag-off one-step jaxprs of the full ORCA2 and DINO cards identical to base `e1b56d6004c6`; V2's and
  V1's card fields and initial state unchanged (their card tests).
