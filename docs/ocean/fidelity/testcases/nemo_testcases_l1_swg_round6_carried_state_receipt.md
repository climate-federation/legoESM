# NEMO SWG level-1 round 6 — whole-step carried state

Date: 2026-10-11. Base: `5b7c5397d`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round6`.

## Preregistration

This section is frozen before any round-6 implementation or measurement. The
operator's independent round-5 review returned SHIP. Round 5 made every
given-entry and chained stage transition bit-identical, while the independent
production ladder first differs at `kt=6` after five exact whole steps.

Frozen predictions and falsifiers:

1. **Complete step-5 carry audit.** Starting only from the SWG card's shipped
   rest state and analytic wind, the production `step()` path will run five
   steps. Its step-5 public after-state and the exact stage-3 output will be
   compared field by field with the admitted `kt=6, stage=1` entry: SSH, U, V,
   before U/V, before/current U/V stresses, `r3t`, `r3u`, `r3v`, and `r3f`.
   The redundant west U and south V storage, masks, and every other SWG state
   leaf will receive an explicit equal/derived/inert disposition. An omitted
   leaf or a comparison made only on the five restart fields **REFUTES** the
   audit.
2. **Boundary prediction.** The step-5 stage-3 output is predicted
   bit-identical to the admitted `kt=6, stage=1` entry on all 13 recorded
   fields, as round 5's chained row already reports. The independently carried
   public SSH/U/V and stress fields are also predicted exact. Any earlier
   unequal field **REFUTES** the proposed boundary and becomes the first
   source-order owner.
3. **F-ratio attribution.** The first reconstructed-entry difference is
   predicted to be `r3f`. NEMO computes the four T-cell numerator with the
   explicitly parenthesized pairwise association and then multiplies by
   `r1_hf_0*r1_e1e2f` at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:218-229`; stage 3 writes that
   ratio at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:336-338`, swaps the after slot
   into `Nbb` at `:358-366`, and the next step records it without recomputing
   at `:147-159`. legoESM's stage output selects source-associated evaluation,
   but `_entry_from_state` currently reconstructs `r3f` without that selector.
   A first difference in any other field, or bit-identical reconstructed
   `r3f`, **REFUTES** this attribution.
4. **Landing arm.** If item 3 is confirmed, the production whole-step entry
   reconstruction will use the same shared source-associated QCO F-ratio
   builder already used by the stage boundary. No new numeric implementation,
   state field, default, or configuration selector will be added. The arm is
   confirmed only if the complete carry audit becomes exact and the
   independently initialized ten-step ladder reaches 50/50 bit-identical
   restart rows. Any remaining row **REFUTES** completion and no 30-day
   acquisition will be written.
5. **Control.** A direct production test will compare the reconstructed
   whole-step entry with the actual stage-3 boundary on a nonzero manufactured
   SSH field whose source-associated and unassociated F-ratios differ. Removing
   the selector must make the test fail. A uniform/zero field or a case where
   both associations happen to be equal **REFUTES** the plant.
6. **Rule 12.** The 338/338 given-entry and 338/338 chained rows may not move,
   and no existing independent AT-BAR row may leave the bar. The only shared
   helper behavior change is an explicit SWG caller opt-in; all other callers
   retain their defaults. TSUNAMI, VORTEX, VORTEX_VEC, GYRE, LOCK_EXCHANGE,
   OVERFLOW, DINO, and ORCA2 will be shown not to execute this `swe_rk3`
   boundary or remeasured. Any regression **REFUTES** landing.
7. **Thirty-day stop condition.** Only after the independent ladder is 50/50
   bit-identical will this round add a passive, admitted 30-day NEMO
   acquisition runner under a new target name, derived from the P3 runner and
   shipped deck. It will print a named `REFUSE` line before every nonzero exit,
   record at least daily comparison boundaries, and require its instrumented
   run's ordinary NEMO outputs to be bit-identical to the admitted P3
   reference. No NEMO executable, `makenemo`, or `mpirun` will run here.
8. The receipt-citation gate must map every compiled-source citation, and its
   shifted-citation plant must exit nonzero. Focused stage, ladder, geometry,
   card, differentiability, and control tests must pass on CPU in fp64. A
   green numerical plant, a shifted citation that passes, or a changed
   certified-card hash **REFUTES** landing.

No configuration or carried-state choice is introduced. The proposed arm
preserves the already selected source association at the next production
boundary rather than adding a state leaf.

## Independent review

Independent review run by the operator.

## Measurement

Preregistration commit: `863a4d77ca4ba32628163b61b8a9e342d7e0646e`.
Committed carry instrument: `984c92d01d5d752f0af2b76a86052d4646305676`.
Clean measured implementation: `84d44ab3840e18302eaf8897bcdcaf98c4b5eb65`.
Thirty-day acquisition tool: `31b8305873efae632dbf7da11d658e9865b3184a`.

1. **CONFIRMED: the whole-step boundary is complete and exact.** The
   independently initialized production run's step-5 SSH, U, V, zero external
   means, previous stresses, all nine static/unchanged array fields, 36
   optional-`None` leaves, and zero redundant west-U/south-V wall storage are
   explicitly disposed. The reconstructed `kt=6` entry is bit-identical on all
   13 admitted fields. Together with the exact stage-3 output, the audit has
   26/26 exact boundary rows. Pre-fix artifact `swg_carry_audit_before.json`,
   SHA-256
   `c32ce198faad06fae96755e0f8528ea4db2d306de371b82f89fc0d9f8e669000`.
2. **CONFIRMED: no public carried state differs.** SSH/U/V match the admitted
   entry after step 5, `uu_b/vv_b` retain their shipped zero values, stresses
   carry the analytic forcing bits, every static field is unchanged, and every
   unselected optional field remains `None`. No state leaf or redundant face
   is unaccounted.
3. **REFUTED: reconstructed `r3f` is not the owner.** The stage-3 `r3f`, the
   value rebuilt from the step-5 public SSH, and the admitted `kt=6` value are
   already bit-identical. NEMO's explicit F-ratio association remains correctly
   transcribed from compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:218-229`, but changing it is neither
   necessary nor eligible. No ratio or state field was added.
4. **CONFIRMED: the first non-bit boundary was compilation across stages.** A
   committed extension ran the exact `kt=6` entry through ordinary `step()`.
   The separately JIT-compiled production stages remained exact, but the old
   monolithic whole-step JIT changed one wet U face `[11,1]` by
   `4.336808689942018e-19`; SSH and V stayed exact. Artifact
   `swg_carry_audit_step6.json`, SHA-256
   `12efa52aadd865fffc4953ec0258fcd041bde6e9793dfd53fae8bccf0710aa70`.
   NEMO stores and boundary-links each stage result before swapping its time
   slots at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:204-210`,
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:278-284`, and
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:360-366`; one fused XLA graph did
   not preserve those three materialization boundaries.
5. **CONFIRMED: stage materialization owns the debt.** Normal shared `step()`
   now dispatches its three stages through the same JIT production-stage
   kernel already certified by the given-entry ladder. The arithmetic, state,
   card, defaults, and selectors are unchanged. A direct plant invokes the old
   monolithic `_step_jitted` on the exact step-5 state and reproduces exactly
   the one changed U value and magnitude above. Removing the staged dispatch
   therefore makes the test fail. Reverse-mode through ordinary `step()` is
   finite and nonzero.
6. **CONFIRMED: the ten-step target is met.** The independent ladder improves
   from 37/50 to 50/50 bit-identical rows; `first_kt_over_bar` is now `None`.
   The final carry audit is 29/29 exact. Given-entry and chained controls remain
   338/338 exact. Artifact SHA-256 values are respectively
   `bef2df374379b32ea11a33e2a068d00bcfb3085571f965a26171a789acdcc0cf`,
   `2bc714f07365a862172609948ad916b7783e57b4fb40e60aae6f1cda4b4ce1c1`,
   `791b54b39c2baf86775f981656987d164f379ded9bcb0001a76d154d5a6aefd9`,
   and `cd8aa01f6caa1b7baeecb4230827042ff68f5e646f182363dbc4b2099dfab11e`.
7. **CONFIRMED: controls bind.** The admitted nonzero-input one-ULP plant
   remains DEBT and exits 1; artifact SHA-256
   `0d70fcf521b71c65cde24cac0bb4eb76d013e22e82a0dee146b2ff233b5620f1`.
   The new monolithic-stage plant changes the scored production U output. The
   geometry signed-y Coriolis, wet-cell, and layer-thickness plants each exit
   1.
8. **CONFIRMED: the 30-day runner is ready, not run.** The committed
   `run_30d.sh` preflight hash-verifies the reference binary, applies an
   additive daily stage-1 writer under new target `SWG_OMIP_L1_P3D`, preserves
   the shipped physical deck while setting only `nn_itend=720` and daily
   output/restart cadence, runs reference and instrumented binaries separately,
   and admits 30 daily entries plus 30 daily restarts only if every ordinary
   grid and restart variable is bit-identical. Its preflight prints
   `PREFLIGHT_OK`; NEMO, `makenemo`, and `mpirun` were not executed here.

## First non-bit statement

There is no differing carried field or NEMO arithmetic statement at the
step-5/step-6 boundary. The first non-bit operation was legoESM's compilation
of all three otherwise exact stages into one XLA graph. The first missing
source semantic was the materialized stage-1 boundary: NEMO boundary-links the
new U/V and ratios and swaps `Naa` into `Nnn` before stage 2 at compiled
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:204-210`. The same materialization
repeats before stage 3 and before the next whole step at compiled
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:278-284` and
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:360-366`. Preserving those boundaries
removes the first observed non-bit value and all propagated debt.

## Rule 12 disposition

No AT-BAR row leaves the bar: given-entry and chained rows remain 338/338
bit-identical, while the independent rows move only from DEBT to exact. A
repository execution search finds that only `build_swg_card()` selects
`outer_integrator="swe_rk3"`; TSUNAMI, VORTEX, VORTEX_VEC, GYRE,
LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 do not execute the new public-step
branch.

The cross-card battery is 273/276 green. Its only failures are the already
documented stale expected digest assertions; the actual GYRE, LOCK_EXCHANGE,
and OVERFLOW digests remain the round-5 values `f02d2f42a570f413`,
`d5dfdf83b3a1e0ab`, and `83de0010e1170032`. No card or shared default changed.

## Gates

The SWG card, stage, differentiability, ladder, geometry, and admission battery
passes 194/194. Geometry remains 34/34 bit-identical; clean artifact SHA-256 is
`ce334fcbf0111e01be356fe5441c2327017cbf4ca6da0af90a802f8638b47cfa`.
The three geometry plants exit 1. The given, chained, independent, and final
carry gates exit 0; the one-ULP input plant exits 1.

The citation gate and its shifted-citation plant are recorded after this
receipt's measurement commit.

## Choices this round

No unasked physical, configuration, threshold, or carried-state choice was
made. The user directed the 720-step duration, daily acquisition boundary,
new target, and passivity criterion. The implementation changes compilation
boundaries only; it adds no state.

## OPEN

1. The operator runs the independent review of this round.
2. After review, the operator runs `run_30d.sh`; the next round admits and
   scores its daily record before proceeding to the shipped 172800-step
   statistics.
