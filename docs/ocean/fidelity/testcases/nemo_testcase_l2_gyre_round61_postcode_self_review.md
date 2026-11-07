# Round 61 post-code self-review — kt=2 carried-operand split

Date: 2026-09-11. This review covers the extension of the existing round-54
reader/walk, its focused test, citation-map additions, and the round-61
receipt. No production physics, card, state schema, harness, reconciliation,
freshwater, or guard file changed.

## Findings dispositioned

1. **The requested 744 baseline is not reproduced.** The actual production
   kt=2 carry scores 5,325 unequal K_H cells. The receipt records this as the
   preregistered stopping boundary; it does not relabel the largest reduction
   as ownership.
2. **Early probe defects were not allowed to become evidence.** Two dry runs
   failed on the entry-avm extent and face-metric extent. The admitted v6
   artifact uses the corrected W-row mask and the compiled qco
   e3w-reference-times-r3 face metrics, and has a clean worktree stamp.
3. **The face-shear claim has an equal-input control.** NEMO Kmm/Kbb
   velocities through the shared production statement reproduce recorded sh2
   in all 17,400 cells. The model has no Kbb face-velocity carry, so only the
   explicitly labeled model-Kmm/NEMO-Kbb arm is reported; no synthetic state
   field is presented as model behavior.
4. **The plant matches the preregistration.** It advances one exact wet
   e3t_Kmm operand by one fp64 ULP and exits nonzero after observing one
   unequal cell. Existing closure-result and producer-stamp plants remain.
5. **Rule 12 stops the campaign here.** No single or cumulative restored-tip
   substitution reaches zero, so there is no eligible statement fix and no
   cross-card trajectory claim to gate. kt3/day-30 values are correctly
   reported as not applicable, not silently reused from the held arm.

## Verification

Focused pytest: 11 passed. Python compile and `git diff --check`: pass.
Citation gate and its shifted-citation plant are run only after this review and
receipt are committed, so their worktree stamps can fail closed.

ASKED: post-code adversarial self-review. UNASKED: no physics/configuration or
state-carry expansion. Open risk: the public closure output remains entangled
with the round-60 held downstream associations; a future attribution must
preregister one same-program baseline rather than combine arms.
