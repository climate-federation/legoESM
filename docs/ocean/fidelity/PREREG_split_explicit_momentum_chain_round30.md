# Preregistration: `dom_qco_r3c` last-bit operand factorial, round 30

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before evaluating any
counterfactual arm. Round 29 is bound at SHA-256
`3cd09d6bf3f7f5bb9ee51705a1e5e8cbf3671ff19cec6ebf407efd9e883d60d5`:
row 2 is at its registered `CEILING`; row 3 is the ordered stop because all
four outputs are `NEAR-CLASS` at normalized RMS `2.69e-16`--`3.32e-16` and
maximum/NEMO-RMS `2.07e-15`--`2.50e-15`.

## Source and frozen inputs

The active formulas are `domqco.F90:153-185`. Their depth operands are built
at `domain.F90:139-160`: T/U/V reference depths are source-ordered vertical
left accumulations through `jpkm1`; F uses `e3f_0 * vmask(i,j,k) *
vmask(i+1,j,k)` and the same left accumulation. Their metric operands are
materialized at `domhgr.F90:144-160`: `e1e2* = e1* * e2*`, followed by stored
reciprocals. Existing full-halo `spg_dump_pssh_final.bin` and
`seq_dump_r3{t,u,v}_aaa_kt00005761.bin` / `seq_dump_r3f_kt00005761.bin` are
the only oracle arrays. No NEMO run or new writer is permitted.

## Registered 2^3 factorial

Score all eight arms on the same four wet populations and unchanged strict
pointwise `1e-15` class bars used in round 29. The axes are:

* **P (SSH):** P0 is the current production row-1.4 `pssh_final`; P1 substitutes
  the existing NEMO `spg_dump_pssh_final.bin` pointwise.
* **D (depth):** D0 retains NumPy vector reductions and the current F-mask
  shorthand; D1 transcribes `domain.F90:139-160` literally, including Python
  source-ordered vertical left accumulation through `jpkm1`, F's two V-mask
  factors, and the materialized reciprocal-depth assignments.
* **M (metric/post factor):** M0 retains live division by reconstructed face
  area; M1 materializes `e1e2t`, `e1e2{u,v,f}` and their reciprocals in the
  `domhgr.F90:144-160` association, then evaluates the U/V/F pair sums and
  ordered `* r1_h0 * r1_e1e2*` post factors exactly as
  `domqco.F90:165-181`.

P1 is an oracle substitution used only to distinguish inherited row-1.4 debt;
it is not an admissible production implementation. D1 and M1 are admissible
literal builders because every operand is already present in `NemoGrid`.

## Bars and disposition

Identity must be `AT BAR`; a four-nextafter plant must fail; zero alignment
must beat all eight neighbouring shifts. Input hashes, full-halo shapes,
populations, CPU/fp64, lane, source excerpts, and tracked-clean state are
mandatory.

For each output report the eight raw metrics, the P/D/M main effects, all
pairwise interactions, and the three-way interaction using normalized RMS
error. Also report fractional removal from P0D0M0; when the baseline is
nonzero, a main effect or interaction is called material only at at least 10%
absolute removal and called an owner only at at least 90% removal. Numerical
signs are descriptive; only the frozen classifications below determine the
walk.

* **LOCALIZED_TO_ADMISSIBLE_QCO_ASSOCIATION:** P0D1M1 makes all four fields
  `AT BAR`. Build that literal depth/metric composition in production and
  re-run row 3 against the unchanged bars.
* **INHERITED_FROM_ROW_1_4:** P0D1M1 is not all `AT BAR`, P1D1M1 is all
  `AT BAR`, and P removes at least 90% of the P0D1M1 normalized RMS in every
  still-failing field. Resume at row 1.4; do not patch QCO around its input.
* **COMPOSED:** neither single admissible axis owns the residue, but P0D1M1 is
  all `AT BAR`; build the complete D+M literal composition and record the
  interaction table.
* **OPEN_UNRESOLVED:** no arm is all `AT BAR`, or a control/provenance receipt
  fails. Stop the ordered walk and design new instrumentation.

Rows 4--6 and every later registry chain remain ordered-blocked until current
production row 3 itself is `AT BAR`.
