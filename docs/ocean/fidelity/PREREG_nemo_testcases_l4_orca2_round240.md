# ORCA2 round 240 preregistration — fold-row HPG operand split

Date: 2026-10-11. Frozen base: `f431cb7bf`. This round executes round 239's
OPEN and operator note B58 addendum 40. No deck, carried state, stabiliser,
sea-ice selector, or `unmeasured_features` entry changes.

Every ORCA2 result is retained separately as **independent** or **given
NEMO's entry**. Execution is CPU, production JIT, fp64/libm. Only admitted
NEMO records, passive completed states, and offline replay may be used; no
in-executable observer is permitted.

## Frozen executing source and order

The resolved deck executes `hpg_sco`. NEMO seeds the meridional along-level
accumulator `zhpj`, forms the local s-coordinate correction `zvap`, and writes
`pvv = zhpj + zvap` at
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-427`; it repeats
that order at interior levels at
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:445-463`. The surface
`zvap` statement reads north/current `rhd`, north/current live `gdept_z0`, and
`r1_e2v` in that order. The live depth is the compiled qco expression
`gdept_1d*(1+r3t)-ssh` visible directly in those lines.

The `zps_hde` partial-step bottom correction is not in this executing branch.
It is therefore a dead-arm hypothesis for round 240 and cannot own or clear an
ORCA2 `hpg_sco` row. No number from another HPG branch is mixed into the table.

Round 239 fixes the target as the 68 active northern-fold V faces. The admitted
round-180/182 records carry exact `zhpj`, `zvap`, `pvv`, north/current `rhd`,
north/current live `e3w`, and the metric for kt=1 stage 1. Round 185 already
landed NEMO's T-fold association for the north density read. Round 238 made the
raw face geometry and masks exact.

## Frozen predictions and falsifiers

- **R240-P1 — admission and calibration.** Both rank records re-admit with
  exactly-once 148x180 coverage and retained restart identity. NEMO's recorded
  operands replay recorded `zhpj`, `zvap`, and `pvv` bit-for-bit. The current
  offline candidate replay reproduces round 239's 68-face HPG endpoint within
  the fixed `2e-10 m s-2` floor. Any missing/malformed stream, moved restart,
  replay miss, stale tree, wrong dtype/backend, or non-firing control REFUTES
  the instrument.
- **R240-P2 — component boundary.** With round 185's north-density association
  present, `zhpj` is predicted at the floor on every contributing fold-row
  level. `zvap` is predicted to be the first non-bit component, and therefore
  `pvv = zhpj + zvap` carries the same effective debt. A non-bit `zhpj`
  REFUTES this prediction and routes the OPEN back to its first primitive
  operand; no `zvap` claim is then made.
- **R240-P3 — primitive operand.** In the executing `zvap` statement, north
  and current `rhd`, current `gdept_z0`, and `r1_e2v` are predicted exact on
  the registered support. The first non-bit primitive is predicted to be the
  north live `gdept_z0` read because the literal operator folds `rhd` but its
  `gdept_z0` north neighbour still comes from a cyclic roll. If another
  primitive is first, retain the count/max/argmax and name that operand instead.
- **R240-P4 — statement and one-variable arm.** The first non-bit executable
  statement is predicted to be the surface north-minus-current live-depth
  difference inside `zvap`. Replacing only the north live-depth operand with
  NEMO's recorded value must close `zvap` and `pvv` at the 68-face endpoint;
  a byte-identical arm or a residual above the floor REFUTES ownership.
- **R240-P5 — conditional production candidate.** Only if P1-P4 confirm, route
  `gdept_z0` through the existing T-fold helper at NEMO's `hpg_sco` program
  point; do not add a second fold implementation. The direct operator must be
  bit-exact given NEMO's inputs. Then re-score the complete private round-217
  unit atomically on OMT-4 and rung 0. The unit stays private if it still
  refuses or if its first-over-bar/majority/SSH predicates fail Decision 96.
- **R240-P6 — landing gates.** A qualifying production statement must pass
  OMT-1..4, rung-0 and rung-10 ladders, the OMT-4 and rung-0 month boundaries,
  GYRE's certified ladder/year, DINO month, DINO/LOCK/OVERFLOW/VORTEX/tank/
  generic-card gates, focused tests, the once-only ocean-fidelity battery,
  both citation gates with a firing rigid-shift plant, and independent
  read-only review. Any non-Decision-96 red blocks landing.

`LANDED` requires P1-P6. `HELD` keeps every failed prediction and names the
first source-ordered HPG statement. `ACQUISITION_NEEDED` is allowed only if an
admitted record lacks an operand the executing branch actually reads.

ASKED choices: Decisions 96, 103, 109, 113, 114 and 115. UNASKED choices:
empty.
