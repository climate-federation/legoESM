# Preregistration: split-explicit chain round 11 — arithmetic association

Date: 2026-08-29. Status: **FROZEN BEFORE MEASUREMENT**.

## Ordered question

Round 10 fixed the V-face Mercator metric and advanced the first row-1.3 debt
to the arithmetic tail:

| subrow | operand | class | fixed-tree max error / NEMO RMS |
|---:|---|---:|---:|
| 9.4 | `zhU` | POINTWISE `1e-15` | 2.07016814618e-15 |
| 9.5 | `zhV` | POINTWISE `1e-15` | 2.67802415708e-15 |
| 9.6 | `zhdiv` | ACCUMULATING `1e-12` | 3.09469003012e-10 |

This round uses only the existing, SHA-bound day-180 dumps. It first tests
9.4. It may test 9.5 only if 9.4 is owned, and 9.6 only if both metric-flux
products are owned. A later calculation is `ORDERED-BLOCKED` if an earlier
row is not resolved.

## Active NEMO expressions and arms

The uninstrumented DINO oracle evaluates:

- `dynspg_ts.F90:699-701`: `zhU = e2u * ua_e * zhup2_e`;
- `dynspg_ts.F90:702-704`: `zhV = e1v * va_e * zhvp2_e`;
- `dynspg_ts.F90:722-724`:
  `zhdiv = ((zhU(i)-zhU(i-1)) + (zhV(j)-zhV(j-1))) * r1_e1e2t`.

For each three-factor product, the scorer compares all three distinct
pair-first associations in CPU/JAX fp64:

1. NEMO literal left association: `(metric * velocity) * depth`;
2. metric-depth first: `(metric * depth) * velocity`;
3. current production association: `(depth * velocity) * metric`.

Candidates that are bit-identical on the registered wet population form one
equivalence class. A row is **LITERAL-OWNED** only if the class containing the
NEMO literal is AT BAR and no bit-distinct class is also AT BAR. Multiple
labels for one bit-identical class are not ambiguity. If a bit-distinct
non-literal class also clears, the result is `AMBIGUOUS`; if the literal class
does not clear, it is `OPEN_UNRESOLVED`.

After U and V are LITERAL-OWNED, divergence is tested from the directly dumped
oracle `zhU/zhV` transports with four candidates:

1. NEMO literal: `(du + dv) * r1_e1e2t`;
2. divide after sum: `(du + dv) / e1e2t`;
3. normalize axes separately: `du*r1_e1e2t + dv*r1_e1e2t`;
4. the current generic velocity-divergence route, which divides oracle
   transports by face widths and multiplies them back inside
   `divergence_cgrid`.

The same equivalence-class ownership rule applies. Periodic U west-neighbours
and closed-wall V south-neighbours follow the production C-grid mapping; exact
wet T/U/V populations remain 9,920/9,758/9,868.

## Gates and authorized fix

Every arm uses the unchanged campaign classifier: correlation >= `1-1e-9`,
mean-absolute ratio within `1e-6`, and max error/NEMO RMS <= `1e-15` for
9.4/9.5 or <= `1e-12` for 9.6.

If and only if all three rows are LITERAL-OWNED, a minimal production fix is
authorized in the NEMO-faithful split-explicit path:

- assemble metric transports in NEMO operand order;
- evaluate their divergence in the NEMO literal subtraction/sum/reciprocal
  order;
- retain masks, closed-wall zeros, periodic U topology, JAX transforms,
  autodiff, and the generic `divergence_cgrid` contract for other callers.

The fix is not accepted from the arm alone. The unchanged round-9 production
scorer must then be rerun on the same six dumps, and 9.4, 9.5, and 9.6 must all
be AT BAR at their original class gates. Otherwise the fix is rejected or the
row remains open.

Only a production rescore with 9.1--9.7 all AT BAR releases row 1.4. Rows 2--6
remain ordered-blocked until row 1.4 itself clears.

## Provenance and controls

The scorer refuses changed round-8/round-10 receipts, dump hashes, NEMO
binary/source/bracket bindings, runtime, timestep, package commit, backend,
fp64 setting, session ID, or dirty tree. Identity must classify AT BAR; a
`1e-6` NEMO-RMS planted cell must classify DEBT through the same classifier;
the fixed-tree production baselines must reproduce the committed row-10
metrics; and the literal/non-literal equivalence partition is recorded rather
than inferred from labels.

No NEMO run/build, new writer, SLOT block, GPU, `mpirun`, or push is
authorized in this round.
