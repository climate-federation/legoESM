# ORCA2 round 238 preregistration — exact face-geometry landing

Date: 2026-10-11. Frozen base: `f9b157da26f29b89c5b14d33c540a2904c1b6020`.
Scope: Decision 114 only. Decision 115 is already landed. The round-217
V-transport unit remains out of production until round 239. No deck, carried
state, stabiliser, sea-ice selector, or `unmeasured_features` entry changes.

Every ORCA2 number is retained separately as **independent** or **given NEMO's
entry**. Production JIT, CPU, fp64 and scalar-libm are mandatory.

## Frozen source unit and implementation

The executing ORCA2 build reads `e3u_0/e3v_0` from its domain file at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90:184-188`; the compiled
substitution header maps `E3u_0/E3v_0` directly to those three-dimensional
arrays at `ORCA2_OMIP_L4_R230FOLDTRP/BLD/inc/domzgr_substitute.h90:114-121`.
NEMO derives U/V/F masks from `tmask` and associates them with sign `+1` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:206-214,228-232`.
For the executing T-pivot fold, the V rule writes pivot and halo rows from
their mirrored sources at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:973-981`; the T/U/F
rules are the adjacent cases at `:945-972,982-990`.

The candidate routes the WS-RK3 stage geometry through the existing
`nemo_qco_resolved_mesh_operands` builder. Cards carrying a raw NEMO mesh use
the attached `e3u_0/e3v_0` and associated masks; cards without one retain the
existing reconstructed source. No second geometry builder or card selector is
allowed. Round 233 already measured the ORCA2 operand endpoint privately:
live `e3v` `13/226,637 -> 0`, `vmask` `668/399,600 -> 0`, while the independent
`vn_adv` debt remained `8,589/8,589`; this round lands the exact geometry and
does not claim that it closes that separate debt.

## Frozen predictions and falsifiers

- **R238-P1 — source and registry.** The production ORCA2/OMT path must use
  the one resolved builder. Raw `e3u_0`, `e3v_0`, `tmask`, `umask`, `vmask`
  and `fmask` must be bit-exact against the admitted NEMO bundle on their
  registered extents. A reconstructed-source, one-bit face-thickness, one-bit
  mask, missing-member, or second-builder plant must refuse.
- **R238-P2 — measured operand boundary.** Under both labels, kt=1 stage-1
  live `e3v` and `vmask` must remain `0` unequal on supports `226,637` and
  `399,600`. The independent external-mode rows are expected to remain debt;
  any claim that geometry alone closes `vn_adv` is REFUTED unless it actually
  reaches zero.
- **R238-P3 — ORCA/OMT registration.** Run OMT-1..4, independent rung 0, and
  given-entry rung 10. Register every moved row, RMS direction, maximum
  direction, first debt, exact-row loss and signed-zero/two-ULP count. The
  cited exact geometry lands by Decision 114 even if it exposes the already
  registered external-mode debt; a lost certified exact row, earlier refusal,
  or new unregistered non-finite value REFUTES the landing.
- **R238-P4 — month boundaries.** OMT-4 must not stop before its admitted 96
  steps. Rung 0 must not stop before round 237's step-100 live-thickness
  refusal. Record later completion or the exact new boundary under the
  independent label; an earlier boundary REFUTES.
- **R238-P5 — preserved cards.** GYRE's certified ladder/year, DINO's month,
  and LOCK_EXCHANGE/OVERFLOW/VORTEX/tank gates must satisfy their standing
  bars. Prediction: their certified trajectories are bit-identical because
  either their raw and reconstructed operands coincide or they do not execute
  the ORCA2 raw-mesh stage path. Any movement is registered, never assumed
  away.
- **R238-P6 — implementation integrity.** Focused tests must fail on the old
  reconstructed face source and on a perturbed fold mask. JIT/autodiff/pytree
  purity remains intact. The default and round citation gates have zero
  unmapped anchors and their rigid-shift plant fires. The once-only
  `tests/ocean/fidelity -n 12` battery and independent read-only review run.

## Landing rule

`LANDED` requires P1/P2/P5/P6, no exact-row loss or earlier boundary in
P3/P4, and complete moved-row registration. Otherwise `HELD` names the first
red predicate. No NEMO acquisition is expected; admitted records cover every
gate.

ASKED choices: Decisions 114 and 115, and standing Decision 96. UNASKED
choices: empty.
