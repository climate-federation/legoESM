# NEMO testcase L2 GYRE higher-resolution round 23 receipt

Date: 2026-10-20. Branch: `fidelity/gyre-high-resolution`.

## Decision

**STOPPED FOR DECISION — the certified-card digest moved across the imported
ORCA2 merge.** The round required an immediate stop if either the shipped GYRE
ladder or digest moved. The first parent of merge `846eee4ae4`,
`7e78361fb1`, pins `GYRE-zco` at `56c5171d43504b54`; the merged tree pins and
computes `dca02dffba3eb14f`. The executable digest regression passes at the new
value. This round does not accept that re-pin, change physics, or start the
closing perturbation experiment.

The imported test explains that the digest movement accompanies appended
shared ORCA2 configuration fields, an explicit `rn_bt_alpha`, and the
card-selected flux-form external-mode switch. That explanation is not a
substitute for the user decision demanded by the round. The exact resolved
GYRE configuration delta has not yet been admitted as inert by this lane.

## Finding -> disposition -> evidence

### 1. Merged-source citation drift -> repaired

All 80 drifted source-map entries were re-read against the merged sources and
updated in commit `8841cb1692`. Nineteen used citations in the phase-3 GYRE
receipt were updated. One ORCA2 citation exposed a substantive source change:
the specialized southern-`ff` helper no longer exists because it was
generalized; its receipt now records the current helper and the later
denominator correction instead of merely renumbering the deleted anchor.

The citation unit suite reports `17 passed`. The real audit reports 274
citations, zero failed or unmapped citations, an empty map audit, and all nine
synthetic controls firing. The planted `stp2d.F90:128` shift exits nonzero with
`SYMBOL-NOT-AT-LINE`, so the repair did not weaken the gate.

### 2. Imported ORCA2 merge -> GYRE digest moved; stop condition fired

The certified-card pin changes in the merge diff as follows:

| boundary | commit | `GYRE-zco` digest |
|---|---|---|
| pre-merge first parent | `7e78361fb1` | `56c5171d43504b54` |
| merged lane base | `846eee4ae4` | `dca02dffba3eb14f` |

The targeted executable digest test passes on the merged tree (`1 passed`),
proving that `dca02dffba3eb14f` is the current computed card digest rather than
stale receipt text. No digest expectation was edited in this round.

The actual one-degree no-worsening test also ran inside the full battery and
passed: the current ten-step residual rows remain at or below all 70 registered
round-9 ceilings. This is useful evidence about the ladder but cannot waive the
explicit independent digest stop condition. A byte-for-byte re-admission and
the closing perturbation check were therefore not run.

### 3. Closing rounding-order check -> deliberately not run

Round 22 leaves a measured stage-1 lateral-diffusion composition residue of at
most `3.3881317890172014e-21` in V and `1.852884572118782e-21` in U. The
requested experiment would preregister and measure whether a perturbation at
that scale reproduces the days-30--180 legoESM self-divergence relative to the
NEMO scalar/vector envelope. Doing so after a certified-card identity stop
would mix two campaign boundaries, so there is no preregistration, GPU request,
or result from that experiment in this round.

## Certification boundary and inherited pins / OPEN

The five-year `r4_smag` climate result at commit `219db484b` remains an
inherited PASS against both registered NEMO builds for its recorded code and
deck. It does not certify the newly merged card digest. Inherited scientific
state remains: 31 levels, WS-RK3, split-explicit free surface, TKE/EVD, Redi
tracers, the literal 360-day forcing clock, the scalar-libm CPU ladder policy,
and the registered stage-1 composition debt. No NEMO source, NEMO run, GPU
run, deck, physics, configuration, or acceptance bar was changed here.

OPEN, in order:

1. User decides whether the merged `dca02dffba3eb14f` card may replace the
   admitted `56c5171d43504b54` card. The recommended path is to admit it only
   after an exact resolved-configuration diff proves every newly carried GYRE
   value inert, then re-run the byte-for-byte ten-step comparison.
2. If admitted, preregister and run the one closing `3.3881317890172014e-21`
   perturbation experiment over the existing days-30--180 metrics.
3. Preserve the inherited five-year climate certification as provenance for
   `219db484b`; do not silently transfer it to the merged implementation.
4. Post the finding to issue #1455 when network access and authorization are
   available.

## Tests and review

The final disjoint GYRE battery is green:

- high-resolution instrument, acquisition, climate, driver, ceiling, citation,
  and namelist-guard split: `173 passed, 2 deselected`;
- time-level, one-degree reconciliation, and phase-3 compatibility split:
  `62 passed, 2 deselected`;
- total: `FULL_LANE_BATTERY_SPLIT_PASS: 235 passed, 4 deselected`.

The citation gate run explicitly on this receipt passes its one compiled-source
citation with zero failures or unmapped entries and all nine controls firing.

Review status is **HELD / UNREVIEWED-BY-GLM**. No physics result was landed and
no closing scientific claim was made without that unavailable review.

Choice audit — **ASKED**: repair all merged-source citations, demonstrate that
their controls still fire, check the imported merge against the certified GYRE
digest and ladder, and stop on movement. **UNASKED**: no digest re-pin, card
acceptance, perturbation acquisition, physics edit, NEMO execution, GPU
execution, or external issue update was made.
