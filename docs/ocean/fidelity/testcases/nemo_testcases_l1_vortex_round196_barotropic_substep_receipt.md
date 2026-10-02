# NEMO testcase fidelity: round 196 / VORTEX round 12 — the barotropic substep record and the dyn_spg_ts walk

**Status: HELD.** Nothing lands in production.  The round-196 record is
admitted and the walk names the barotropic Coriolis trend as the growing
owner of the kt=1 end-of-window velocity residual `1.2459e-08` — the same
number rounds 194-195 measured downstream — with the loop-entry forcing
ruled out by substitution and the recurrence ruled out by a conditioning
control.

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round196/`.
Frozen preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_round196.md`,
committed as `a3064b93a` before any measurement.
Decision 82 (user): round 194's two-solve candidate stays HELD, the
barotropic solve is walked first, the two-ULP ratchet is unchanged.

## Compiled program

The RK3 step solves the external mode once per baroclinic step, in
`stp_2D`, and the stages then replace the depth mean of their own updated
velocity with its result. Round 195 proved that handing legoESM NEMO's
recorded barotropic quintuple takes the stage-2/3 output velocity from
`1.2459e-08` to `1.11e-16`, so the owner of what round 194's held candidate
leaves behind is inside `dyn_spg_ts` — but no record carried that routine's
substep operands, and the statement could not be named.

Inside `dyn_spg_ts` the sub-time-step loop runs `icycle` times
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:359`). Each pass
executes, in this order: the AB3-AM4 mid-step extrapolation of the
barotropic velocity (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:389`) and of the sea surface
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:401`); the mid-step face depths (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:414`); the
mid-step transports (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:434`); the after-SSH
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:454`); the running transport sum (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:463`); the
after-SSH at velocity points (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:478`); the half-step-back
interpolation (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:489`, `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:491`); the surface
pressure gradient (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:498`); the barotropic Coriolis trend
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:503`); the explicit bottom stress (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:517`);
the vector-form velocity update (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:535`); the face-depth
update (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:581`); and the time-filter sums
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:628`). After the loop the sums are divided by the weight
sums (`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:648`) and the filtered quintuple is what the stages
consume.

## The record

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/dynspg_ts_substep_record.patch`
adds write-only calls at those boundaries and nothing else: the patch's only
removal line is its own `---` file header, which the harness checks before
it builds. The writer
(`vortex_r12_spgts_terms.F90`) holds no reference to a NEMO field; every
array arrives as an `INTENT(in)` argument and is written after the compiled
statement that produced it has completed.

One stream per baroclinic step, in the campaign's self-describing format: a
16-character magic, fifteen header integers with the 64-bit word size last,
then `(name, rank, n1, n2, n3)` groups with their payloads to end of file.
Group names carry their frame — `i000_` the loop entry, `jNNN_` substep
`NNN`, `o000_` the loop exit — and the number of substep frames is a
function of the header's own `icycle`, so nothing predicts a size (operator
note BD).

## Record admission

Both configurations were built and run here, one `mpirun` at a time, on the
otherwise idle host. The record is ADMITTED
(`phase3/round196/oracle_spgts_substeps/vortex_round196_spgts_admission.json`):

* ten records, `kt = 1..10`, one per baroclinic step, 48.5 MB each;
* each carries 50 frames — the loop entry, 48 sub-time-steps and the loop
  exit — and 1562 groups, and every group's payload matched its own
  declared extents;
* the header's own `icycle` is 48 on all ten and the checker derived the
  frame count from it, so no size was predicted anywhere;
* `"restart_byte_identical": true`.

**Additions-only proof.** The step-10 restart is byte-identical across all
four runs of this card — this round's instrumented run, this round's
uninstrumented reference, and round 192's two runs:
`f09be03c39a877b2dcbc731f…` in every case. The writer changes no answer,
and it is the same card the rounds 192-195 receipts measured.

**Plants.** All four fire, and the unplanted run is green:

| plant | the checker's own refusal |
|---|---|
| `field-name` | `missing group(s) ['j001_zhU']` |
| `truncated` | `group 'o000_ssh_aa' declares 4489 doubles, but only 4488 remain` |
| `missing-frame` | `missing group(s) ['j048_ua_ext', 'j048_va_ext', …]` |
| `header` | the corrupted extent makes the group walk read a payload as a name |

One of the unit controls caught a real defect before the record existed:
the group walk rejected the rank-1 groups the barotropic record uses for
the time-filter weight vectors. Rank 1 is now accepted for this family
only, so the older families' guard is not weakened.

## The substep walk

Both sides are the same card, the same production-jitted `model.step`,
seeded from NEMO's own recorded kt=1 step entry; the comparison is NEMO's
recorded operand against the one legoESM materialises at the same compiled
boundary, for all 48 substeps.

**Every scalar coefficient is bit-identical.** 336 rows — the three AB3-AM4
mid-step extrapolation coefficients and the four half-step-back
interpolation coefficients at each of the 48 substeps — agree to the last
bit, so neither the startup ramp nor the weight schedule is in question.

**Every boundary is at the compiled-rounding floor through substep 25.**
Normalized max abs, `VORTEX_VEC-zco`, kt=1. The campaign's normalisation
divides by `max(peak |NEMO|, 1)`, so for the velocities and the sea surface
it is a relative error and for the trends — whose peaks are far below one —
it is the absolute difference; the budget table further down is therefore
in absolute units throughout, and trend rows must not be compared against
velocity rows without that in mind.

| boundary | j001 | j024 | j027 | j048 |
|---|---:|---:|---:|---:|
| loop-entry `ssh_frc` | `0` (bit) | — | — | — |
| loop-entry `zu_frc` | `2.711e-20` | — | — | — |
| mid-step velocity | `5.551e-17` | `1.665e-16` | — | `9.996e-09` |
| mid-step face depth | `0` (bit) | `1.819e-16` | — | `5.601e-12` |
| mid-step transport | `4.101e-16` | `6.144e-16` | — | `4.121e-08` |
| after-SSH | `9.992e-16` | `2.387e-15` | — | `3.709e-08` |
| surface pressure gradient | `3.320e-19` | `9.216e-19` | — | `6.065e-12` |
| Coriolis trend | `3.388e-21` | `1.216e-19` | — | `7.237e-11` |
| updated velocity | `5.551e-17` | `1.110e-16` | `6.447e-16` | `1.246e-08` |

The updated velocity sits on the floor (`5.6e-17 … 1.1e-16`) for 26
substeps and then grows about threefold per substep, reaching
`1.2458906277138998e-08` at the last substep — **the same number, to
thirteen digits, that rounds 194 and 195 measured for the stage velocity
residual the held candidate leaves behind** (`1.24589062771391176e-08`).
The barotropic solve's own end-of-window output is that residual.

## The residual is not amplified entry error

A walk that finds every boundary at the rounding floor and the loop's
output at `1e-08` has two readings: the loop amplifies what it is handed,
or a statement inside it is wrong. Two controls settle it.

**Conditioning (legoESM against itself).** Perturb the barotropic entry
velocity by one unit in the last place at **every** wet face — 3660 faces,
the same spatial extent the real difference reaches — and re-run the same
production-jitted step. The response stays on the floor for all 48
substeps: `5.551e-17` at substep 1, `1.110e-16` at substep 48,
amplification **2.0** (`phase3/round196/one_ulp_entry_probe.json`). A
single-cell probe was run first and gave amplification 5.0; it is reported
only as a note, because it spreads far more slowly than the real difference
and could not have reached the state the real difference is in when it
starts to grow. The loop does not amplify a last-bit entry change by `1e+08`.

**Substitution (one variable).** Hand the loop NEMO's own recorded slow
forcing at the boundary where legoESM forms it — after the barotropic
Coriolis subtraction — and change nothing else. The loop-entry forcing rows
become bit-exact (`2.711e-20 -> 0`) and the last substep's velocity is
**unchanged in the first thirteen digits**: `1.2458906277138204e-08`
against production's `1.2458906277138998e-08`. The slow forcing is not the
owner.

So the owner is inside the sub-time-step loop.

## The named statement

Decompose the compiled velocity update
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:535`,
`ua_e = ( un_e + rDt_e * ( zu_spg + zu_trd + zu_frc ) ) * ssumask`) into the
differences its own operands carry, with `rDt_e = 60 s`:

| substep | entering velocity | `rDt_e * d(spg)` | `rDt_e * d(trd)` | predicted | measured | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `5.551e-17` | `1.99e-17` | `2.03e-19` | `7.56e-17` | `5.551e-17` | 0.73 |
| 24 | `1.110e-16` | `5.53e-17` | `7.30e-18` | `1.74e-16` | `1.110e-16` | 0.64 |
| 28 | `6.447e-16` | `6.54e-17` | `1.71e-15` | `2.42e-15` | `2.313e-15` | 0.95 |
| 48 | `8.234e-09` | `3.64e-10` | `4.34e-09` | `1.29e-08` | `1.246e-08` | 0.96 |

From substep 28 the budget closes to within 4-5%, so the update statement
itself is faithful and the growth is in a term it reads. That term is the
**barotropic Coriolis trend**, formed at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:503`
(`CALL dyn_cor_2D( ua_e, va_e, zu_trd, zv_trd )`; the explicit bottom
stress at `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:517` adds exactly nothing on this card — the
Coriolis and the post-stress trend rows are identical to the last bit at
every substep). At the last substep it supplies `4.34e-09` of the
`4.70e-09` injected per substep, 92% of it, against the surface-pressure
gradient's `3.64e-10`.

The quantitative signature, and the reason the Coriolis trend rather than
its operand is named: for the first sixteen substeps the ratio of the
Coriolis-trend difference to the velocity difference it is handed is
`6.1e-05 … 8.1e-05`, which is the Coriolis parameter on this beta-plane —
exactly what a faithful operator produces. From substep 24 that ratio rises
to `1.1e-03`, and by substep 32 to `3.1e-02`, 380 times the Coriolis
parameter. Meanwhile the velocity difference the operator is handed is flat
on the rounding floor until substep 27.

**CONFIRMED:** the loop-entry forcing is not the owner (substitution); the
recurrence does not amplify a last-bit entry change (conditioning control);
the velocity update is faithful (budget closes at 0.95-0.96); the Coriolis
trend carries 92% of the per-substep injection at the end of the window.
**PLAUSIBLE, not confirmed:** that the barotropic Coriolis operator itself
differs. The trend-to-velocity ratio is a ratio of two field maxima that
need not occur at the same cell, and no arm has yet substituted NEMO's
recorded per-substep Coriolis trend. That substitution is the closing test
and the record this round built is what makes it possible — see OPEN.

## Predictions (frozen before measurement), kept with their verdicts

* **P1 — the record admits.** **CONFIRMED.** Ten records, 50 frames and
  1562 groups each, `icycle` 48 read from the header, all four plants fire
  and the unplanted run is green.
* **P2 — the instrument is additive.** **CONFIRMED.** The step-10 restart
  is byte-identical to the uninstrumented reference and to round 192's two
  runs of the same card.
* **P3 — the loop-entry operands are not the owner.** **CONFIRMED**, and by
  a stronger test than the one preregistered. The stated falsifier was a
  first-substep extrapolation residual above `1e-12`; the measured value is
  `5.551e-17`, at the floor. The substitution arm then showed that
  replacing the entry forcing outright leaves the end of the window
  unchanged in thirteen digits.
* **P4 — the owner is one statement, re-made every substep.** **REFUTED as
  stated, and the refutation is the finding.** No boundary is non-bit at
  substep 1 and bit-exact later; instead every boundary sits on the floor
  for 26 substeps and then grows together. The owner is not a fixed
  per-substep offset but a term whose difference grows — the Coriolis
  trend, whose ratio to its own operand departs from the Coriolis parameter
  from substep 24 onward.
* **P5 — nothing lands unless a cited statement closes every gate.**
  **CONFIRMED.** Nothing in production changed.

## Gates and tests

* **Production diff is empty.** `git diff` of this round's first commit
  against its last, restricted to `packages/` and `src/`, is zero lines.
  No card, no model file, no certified number can move; GYRE, DINO, the
  tanks and the generic NEMO-GYRE recipe are untouched by construction.
  The DINO month gate is therefore auto-skipped by `land.sh`, as it was in
  round 195.
* **Citation gate** on this receipt and its planted control: quoted with
  the landing below.
* **Focused battery**, run serialized after confirming no other pytest was
  on the host: the citation-gate tests, the NEMO recipe and TKE tests, the
  freshwater closure, the mask-rank and prognostic-barotropic-state tests,
  and the four VORTEX walk-script test modules including this round's own.
  Decisive line quoted with the landing.
* **Record provenance.** `legoesm_git_sha.txt`, `toolchain.sha256`,
  `binaries.sha256`, `shipped_case.sha256` and
  `vortex_round196_spgts_outputs.sha256` are all written into the evidence
  directory by the harness.

## Landing verdict: HELD (record + walk)

Decision 82 stands: round 194's two-solve candidate is not landed, and the
two-ULP ratchet is unchanged. This round lands the preregistration, the
acquisition tool, the per-substep walk with its two controls, the unit
controls and this receipt. No production code changed.

## Option choices made this round

| choice | ASKED / UNASKED | note |
|---|---|---|
| every NEMO deck option | ASKED | unchanged from rounds 3-11; Decisions 69, 70, 73, 75 |
| which boundaries the record writes | n/a | instrument choice, listed in the frozen preregistration before measuring |
| rank-1 groups accepted for this record family | n/a | a reader fix, no model behaviour |

Nothing on the UNASKED list.

## OPEN — round 197

1. **Close or clear the Coriolis trend.** The record now carries NEMO's
   `cor_u`/`cor_v` at every substep. Substitute them, one variable, and
   measure the end-of-window velocity: if it goes AT-BAR the barotropic
   Coriolis operator is the statement and the fix is cited there; if it
   does not, the remaining term is the entering velocity and the owner is
   the recurrence's own composition. This needs a per-substep override
   inside the compiled scan, which does not exist yet — it is the round's
   one new instrument.
2. Re-test round 194's held candidate once the barotropic output is
   AT-BAR, under the unchanged gates (Decision 82).
3. The flux card's (`VORTEX-zco`) own kt=2 owner is untouched since round
   4; Decision 74's 30/15/10-km ladder stays blocked behind the vector
   kt=2 rows.
