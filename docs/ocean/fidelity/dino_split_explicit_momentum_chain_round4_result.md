# DINO split-explicit / momentum-commit chain: round-4 result

Date: 2026-08-29. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Clean measurement commit:
`6d435fe1f53`.

## Verdict

The U forcing residual is a **CONFIRMED_COMPOSITE_OWNER**, not a lateral-
friction-only defect.  The complete `L+V+C` oracle substitution removes
`0.978442231409` of the faithful U residual, has prediction correlation
`0.999779920403`, gain `0.998831790898`, and leaves normalized squared
residual energy `Q=4.647373866160e-4`, clearing every frozen group-ownership
axis.  Here `L` is NEMO's dumped Kbb lateral-friction term, `V` its total EEN
vorticity term, and `C` its pre-loop 2-D Coriolis removal.

All five named noncandidate U terms and the explicit faithful assembly
remainder are individually bounded below `0.10*RMS(r0)`.  U row 1.1 is
therefore fully disposed.  This does not authorize a production edit: the
three terms are a cancelling composition and must be held together in any
future counterfactual.

V does not close under the same trio.  `L+V+C` removes only
`0.091570022652`, with correlation `0.422874223314`, gain
`0.367806569836`, and `Q=0.825245023744`; its Shapley values are descriptive
and explicitly unadmitted.  Hydrostatic pressure gradient is the first and
only unbounded noncandidate V term (`gain=0.910571701895`).  Row 1.1 remains
open for a source-ordered V continuation over `L+V+C+P`; rows 1.2--6 remain
blocked pending that preregistered matrix.

## U composition and interaction table

The matrix reuses the basin reconciliation's EEN × bridge-Omega four-corner
precedent (`19acf16b2f0` preregistration, `820e3500bf3` scorer), extended to
all eight three-axis corners.  All seven locked U corners reproduced within
`1e-12` before `111` was admitted.

| Arm | Oracle substitutions | RMS removal |
|---|---|---:|
| `000` | none | `0` |
| `100` | lateral friction | `0.661241504531` |
| `010` | vorticity | `-3.965153851081` |
| `001` | pre-loop Coriolis | `-3.852810052520` |
| `110` | lateral + vorticity | `-3.806762177948` |
| `101` | lateral + Coriolis | `-3.835115187423` |
| `011` | vorticity + Coriolis | `0.086523828784` |
| `111` | lateral + vorticity + Coriolis | **`0.978442231409`** |

Squared-residual-energy decomposition:

| Quantity | Value | Disposition |
|---|---:|---|
| lateral main | `+0.885242681748` | material |
| vorticity main | `-23.652752764905` | material adverse |
| Coriolis main | `-22.549765405835` | material adverse |
| lateral × vorticity | `+0.662547447803` | material |
| lateral × Coriolis | `-0.713816151557` | material adverse |
| vorticity × Coriolis | `+46.368079455360` | **material cancellation** |
| triple interaction | `0` | bounded |

The Shapley ownership allocation is lateral friction
`+0.859608329870` (`OWNED_POSITIVE_CONTRIBUTOR`), vorticity
`-0.137439313324` (`OWNED_CANCELLER`), and pre-loop Coriolis
`+0.277366246067` (`OWNED_POSITIVE_CONTRIBUTOR`).  These sum to the full
benefit `0.999535262613` within `1e-12`.

Every field-level pair and triple interaction contrast is below
`7e-18*RMS(r0)`, far inside the registered `1e-12` bar.  Thus substitution
commutes exactly at the forcing-field level; the large energy interactions
are the RMS reducer exposing cancellation, not nonlinear model physics.

## Term disposition and active NEMO sources

| Component/term | Active NEMO source | U disposition | V disposition |
|---|---|---|---|
| kinetic-energy gradient | `dynadv.F90:89-95`; `stpmlf.F90:309-314` | BOUNDED_MAGNITUDE | BOUNDED_MAGNITUDE |
| vertical advection | `dynadv.F90:97-103`; `dynzad.F90:81-119`; `stpmlf.F90:309-314` | BOUNDED_MAGNITUDE | BOUNDED_MAGNITUDE |
| total EEN vorticity | `dynvor.F90:143-194`; `stpmlf.F90:315-318` | OWNED_CANCELLER | UNRESOLVED_IN_COMPOSITE |
| lateral friction | `dynldf.F90:73-115`; `stpmlf.F90:319-322` | OWNED_POSITIVE_CONTRIBUTOR | UNRESOLVED_IN_COMPOSITE |
| hydrostatic pressure gradient | `dynhpg.F90:348-413`; `stpmlf.F90:324-328` | BOUNDED_MAGNITUDE | **UNBOUNDED_REMAINDER** |
| pre-loop 2-D Coriolis removal | `dynspg_ts.F90:358-370` | OWNED_POSITIVE_CONTRIBUTOR | UNRESOLVED_IN_COMPOSITE |
| baroclinic-residual drag | `dynspg_ts.F90:372-400` | BOUNDED_MAGNITUDE | BOUNDED_MAGNITUDE |
| centred wind | `dynspg_ts.F90:423-459` | BOUNDED_MAGNITUDE after #1695 | BOUNDED_MAGNITUDE |
| faithful assembly remainder | exact signed ledger remainder | BOUNDED_MAGNITUDE (`1.58e-10`) | BOUNDED_MAGNITUDE (`9.97e-12`) |

The all-eight-term oracle endpoint is NEAR-CLASS for both components, with
normalized RMS error `3.22e-16` U and `2.97e-16` V.  It is retained as a
held-forcing endpoint, not mislabeled AT-BAR under the pointwise arithmetic
class.

## Ordered chain state

| Row | Status after round 4 |
|---:|---|
| 1.1 U | **FULLY_DISPOSED_COMPOSITE_LDF_VOR_COR** |
| 1.1 V | **OPEN_LDF_VOR_COR_PLUS_PRESSURE** |
| 1.2 | ORDERED_BLOCKED |
| 1.3 | ORDERED_BLOCKED |
| 2 second `div_hor` | ORDERED_BLOCKED |
| 3 second `dom_qco_r3c` | ORDERED_BLOCKED |
| 4 `dyn_zdf` | ORDERED_BLOCKED |
| 5 second `wzv` | ORDERED_BLOCKED |
| 6 `mlf_baro_corr` | ORDERED_BLOCKED |

## Controls and attempt audit

The accepted run used only existing day-180 dumps on CPU/fp64.  Locked-
corner, field-linearity, ownership-label, group CONFIRM/REFUTE, and term-bound
plants all fired; inherited round-2 ledger and classifier controls remained
green.  The clean tree was unchanged before/after.

Attempt 1 was invalid before matrix construction because a mutated metric
dict prevented total-field lookup.  Attempt 2 completed but was withheld when
an internal audit found V ownership labels were not scoped to group
confirmation.  Attempt 3 invalidated the preregistered operation-count
roundoff premise for faithful term closure.  The executable retraction makes
that discrepancy an explicit bounded `assembly_remainder`; only the final
clean rerun is admitted.

No NEMO run, writer, integration, SLOT block, GPU, or MPI process was used.

Machine receipt:
`docs/ocean/fidelity/dino_split_explicit_momentum_chain_round4_artifact.json`.
