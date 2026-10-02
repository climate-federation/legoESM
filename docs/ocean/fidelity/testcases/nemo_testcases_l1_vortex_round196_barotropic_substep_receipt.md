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

**Plants.** All four fire and the unplanted run is green. They exercise
THREE distinct guards, not four: `field-name` and `missing-frame` both trip
the required-group cross-check, though `missing-frame` is the only one that
proves the required-frame list is derived from the header's `icycle` rather
than from the file's own contents.

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

**Every boundary stays within a few multiples of the bar through substep
25, and then they all grow together.** To be exact, because an earlier
draft of this receipt said "at the compiled-rounding floor" and the
independent reviewer refuted it: the bar is `1e-15`, 503 of the walk's 1059
rows are DEBT, the first of them is the after-SSH at substep 2
(`1.2768e-15`, 1.28x bar), and the largest normalized residual anywhere in
substeps 1-25 is `6.217e-15`, 6.2x bar. Nothing in that range is bit-exact
and nothing in it is more than an order of magnitude from the bar; what
changes at substep 26 is that the rows start multiplying.
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

The updated velocity sits at `5.6e-17 … 1.1e-16` for 26 substeps and then
grows, by a geometric mean of **2.17x per substep** between substeps 28 and
48 (3.36x at the start of that range, 1.51x at the end), reaching
`1.2458906277138998e-08` at the last substep — **the same number, to
fourteen significant digits, that rounds 194 and 195 measured for the stage
velocity residual the held candidate leaves behind**
(`1.24589062771391176e-08`; relative difference `9.6e-15`). The barotropic
solve's own end-of-window output is that residual.

## The residual is not amplified entry error

A walk that finds every boundary at the rounding floor and the loop's
output at `1e-08` has two readings: the loop amplifies what it is handed,
or a statement inside it is wrong. Two controls settle it.

**Conditioning (legoESM against itself).** Perturb the barotropic entry
velocity by one unit in the last place at **every** wet face — 3660 faces,
the same spatial extent the real difference reaches — and re-run the same
production-jitted step. The response stays on the floor for all 48
substeps: `5.551e-17` at substep 1, `1.110e-16` at substep 48,
amplification **2.0** (`phase3/round196/one_ulp_entry_probe.json`), with
the response reaching 2725-3643 of those faces at every substep, so the
perturbation is not being discarded. A single-cell probe was written first
and discarded as the wrong control — it spreads far more slowly than the
real difference — and its number is deliberately not quoted here, because
no committed artefact carries it. The loop does not amplify a same-sign
last-bit entry change by `1e+08`. This control bounds the response to one
SHAPE of perturbation; the substitution arms above are what rule the entry
out for the actual difference.

**Substitution, operand 1 of 2 (one variable).** Hand the loop NEMO's own
recorded slow forcing at the boundary where legoESM forms it — after the
barotropic Coriolis subtraction — and change nothing else. The loop-entry
forcing rows become bit-exact (`2.711e-20 -> 0`) and the last substep's
velocity is **unchanged in the first thirteen digits**:
`1.2458906277138204e-08` against production's `1.2458906277138998e-08`.

**Substitution, operand 2 of 2 (one variable).** Start the loop from NEMO's
own recorded barotropic entry velocity instead of legoESM's carried pair,
and change nothing else. The arm is self-checking and the check passes: the
first substep's entering velocity rows go from 120 unequal faces to **0**,
so the substitution bound. The last substep's velocity is again
**unchanged**: `1.2458906277138840e-08`
(`phase3/round196/spgts_walk_kt1_nemo_entry_velocity.json`).

Both operands the barotropic loop is handed have now been replaced by
NEMO's own, one at a time, and neither moves the end of the window. **The
owner is inside the sub-time-step loop.**

## What the growth is carried by, and what is not shown

Decompose the compiled velocity update
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:535`,
`ua_e = ( un_e + rDt_e * ( zu_spg + zu_trd + zu_frc ) ) * ssumask`) into the
differences its own operands carry, with `rDt_e = rn_Dt / nn_e = 2880/48 =
60 s`. The table is arithmetic on field MAXIMA, not a per-cell residual —
the walk stores maxima only — so it bounds the budget rather than closing
it cell by cell. The fourth operand, `rDt_e * d(zu_frc)`, is `1.6e-18` and
is omitted from the columns as immaterial.

| substep | entering velocity | `rDt_e * d(spg)` | `rDt_e * d(trd)` | predicted | measured | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `5.551e-17` | `1.99e-17` | `2.03e-19` | `7.56e-17` | `5.551e-17` | 0.73 |
| 24 | `1.110e-16` | `5.53e-17` | `7.30e-18` | `1.74e-16` | `1.110e-16` | 0.64 |
| 28 | `6.447e-16` | `6.54e-17` | `1.71e-15` | `2.42e-15` | `2.313e-15` | 0.95 |
| 48 | `8.234e-09` | `3.64e-10` | `4.34e-09` | `1.29e-08` | `1.246e-08` | 0.96 |

From substep 28 the budget closes to within 4-5%, and the term that carries
the growth is the **barotropic Coriolis trend**, formed at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:503`
(`CALL dyn_cor_2D( ua_e, va_e, zu_trd, zv_trd )`). At the last substep it
supplies `4.34e-09` of the `4.71e-09` injected per substep, 92% of it,
against the surface-pressure gradient's `3.64e-10`. The explicit bottom
stress at `dynspg_ts.f90:517` is active on this card, but it contributes
nothing to the DIFFERENCE: the Coriolis and post-stress trend rows are
identical to the last bit at all 48 substeps.

**Whether the Coriolis operator itself differs is NOT shown, and an earlier
draft of this receipt overstated it.** That draft compared the Coriolis
trend difference against the ENTERING velocity difference; the operator is
handed the MID-STEP extrapolated velocity set at `dynspg_ts.f90:389`, and
the independent reviewer refuted both the operand and the quoted numbers.
Re-measured against the right operand, and per cell rather than
max-against-max:

| substep | faces where the operand differs | median ratio | 95th pct | max | max/max |
|---:|---:|---:|---:|---:|---:|
| 8 | 1775 | `2.79e-04` | `1.29e+00` | `2.44e+00` | `8.14e-05` |
| 16 | 2447 | `2.44e-04` | `2.59e-01` | `9.30e-01` | `8.14e-05` |
| 24 | 3047 | `2.44e-04` | `7.54e-02` | `3.64e-01` | `7.30e-04` |
| 32 | 3417 | `2.10e-04` | `3.34e-02` | `1.09e+00` | `2.16e-02` |
| 48 | 3638 | `1.84e-04` | `1.17e-02` | `8.98e-01` | `7.24e-03` |

The Coriolis parameter on this card (`rn_ppgphi0 = 38.5`) is `9.08e-05`.
The MEDIAN face therefore sees `2.0 … 3.1` times `f`, which is what a
four-triad energy-and-enstrophy stencil gives, and that median is flat
across the whole window: in the bulk of the domain the operator responds to
the difference it is handed exactly as a faithful operator would. The
departure is in the TAIL, and it shrinks as the window runs (95th
percentile `1.29` at substep 8, `1.17e-02` at substep 48) — the opposite of
what a growing operator error would do.

Against that, the reviewer's norm argument: `dyn_cor_2D_init` freezes the
four barotropic Coriolis coefficients, so the operator is a time-invariant
linear map and `max|d(trd)| <= ||L||_inf * max|d(v)|` with `||L||_inf` of
order `f`, with no co-location assumption. The measured ratio of maxima at
substep 48 is `7.24e-03`, about 80 times that bound. The two readings
disagree — a flat per-cell median at `2f` and a max-over-max 80 times `f` —
which means the difference field has become concentrated at a few faces
where the response is far larger than `f`, and the honest statement is that
the walk has localised the growth to the Coriolis trend's few worst faces
without showing that the operator which produces them differs from NEMO's.

**CONFIRMED:** neither loop-entry operand is the owner (two independent
substitutions, each self-checking); the recurrence does not amplify a
same-sign last-bit entry change; every extrapolation, interpolation and
filter coefficient is bit-identical at all 48 substeps; the velocity
update's budget on field maxima closes to 4-5% from substep 28; the
Coriolis trend carries 92% of the per-substep injection at the end of the
window; the end-of-window velocity is the certified downstream residual to
14 significant digits. **NOT SHOWN:** that any single compiled statement
inside `dyn_spg_ts` differs. The closing test is named in OPEN.

## Predictions (frozen before measurement), kept with their verdicts

* **P1 — the record admits.** **CONFIRMED**, with one correction to the
  prediction's own wording: the four plants exercise three distinct guards,
  not four. Ten records, 50 frames and 1562 groups each, `icycle` 48 read
  from the header, every plant fires and the unplanted run is green.
* **P2 — the instrument is additive.** **CONFIRMED.** The step-10 restart
  is byte-identical to the uninstrumented reference and to round 192's two
  runs of the same card.
* **P3 — the loop-entry operands are not the owner.** **CONFIRMED**, and by
  a stronger test than the one preregistered. The stated falsifier was a
  first-substep extrapolation residual above `1e-12`; the measured value is
  `5.551e-17`, at the floor. The substitution arm then showed that
  replacing the entry forcing outright leaves the end of the window
  unchanged in thirteen digits.
* **P4 — the owner is one statement, re-made every substep.** **REFUTED,
  and the refutation is the finding.** No boundary is non-bit at substep 1
  and bit-exact later; instead every boundary stays within a few multiples
  of the bar for 26 substeps and then grows together at 2.17x per substep.
  The owner is not a fixed per-substep offset. The Coriolis trend carries
  92% of what is injected at the end, but per cell its response to the
  operand it is handed is a flat `2 … 3` times the Coriolis parameter
  across the whole window, so this round does NOT name a differing
  statement — it localises the growth and leaves the naming to the
  substitution in OPEN.
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
* **Measurement provenance.** All three arms were re-run on a clean
  worktree at commit `8153857a3a30f24686a7184a038a390fcdcc7c9c` and every
  number in this receipt is from that run: the production walk
  (`spgts_walk_kt1.json`), the NEMO-forcing arm
  (`spgts_walk_kt1_nemo_forcing.json`) and the conditioning control
  (`one_ulp_entry_probe.json`). None carries a `-dirty` stamp.
* **Record provenance.** `legoesm_git_sha.txt`, `toolchain.sha256`,
  `binaries.sha256`, `shipped_case.sha256` and
  `vortex_round196_spgts_outputs.sha256` are all written into the evidence
  directory by the harness.

## Independent adversarial review (fresh reviewer, this round)

Codex is out of budget on this account and the GLM tool is unreachable in
this session, so the mandatory second opinion was a fresh reviewer agent
given the diff, the evidence root and the claims, with no knowledge of how
they were produced. Its verdict was **DO-NOT-SHIP on the receipt text,
ship everything else**, and every finding is accepted and fixed above.

* **BLOCKER (accepted, corrected).** The conditioning control perturbs 3660
  faces, not 3639.
* **BLOCKER (accepted, corrected).** "Every boundary is at the
  compiled-rounding floor through substep 25" contradicted the walk's own
  bar: 503 of 1059 rows are DEBT and the first is the after-SSH at substep
  2. The claim is now the measured one — within 6.2x of bar through
  substep 25.
* **BLOCKER (accepted, corrected, and it changed the verdict).** The
  Coriolis-ratio argument used the entering velocity; the operator is
  handed the mid-step extrapolated velocity. Re-measured against the right
  operand AND per cell, the median response is a flat `2 … 3` times the
  Coriolis parameter across the window, so the earlier reading that the
  operator itself differs is RETRACTED. The reviewer's own norm argument
  (the coefficients are frozen at `dyn_cor_2D_init`, so the operator is a
  time-invariant linear map bounded by `||L||_inf ~ f`) points the other
  way at the field maxima; both readings are now stated, and the
  disagreement is recorded as the thing the OPEN item measures.
* **DEFECT (accepted, corrected).** The single-cell probe's number had no
  committed artefact; it is no longer quoted.
* **DEFECT (accepted, corrected).** The budget table is arithmetic on field
  maxima, and omitted the statement's fourth operand; both are now said.
* **DEFECT (accepted, corrected).** Two of the four plants trip the same
  guard.
* **DEFECT (accepted, fixed in the tests).** The unit controls used a
  square test plane, so a transpose could not fail them, and asserted
  shapes only. The plane is now 8x7 and the controls compare values.
* **DEFECT (accepted, corrected).** "Bottom stress adds exactly nothing" is
  only shown for the difference; and the growth is 2.17x per substep, not
  threefold.
* The reviewer independently CONFIRMED: the patch is additions-only (69
  added lines, 11 hunks, one removal line and it is the `---` header); the
  restart sha256 identical across all four runs; the admission's 1562
  groups `= 21 + 48x32 + 5` and the frame map derived from `header[9]`;
  `scalar_non_bit` empty over 336 rows; the whole budget table reproducing
  with `rDt_e = 60 s`; the headline matching round 195 to 14 significant
  digits rather than the 13 claimed; the forcing substitution landing at
  exactly the boundary claimed; the conditioning control being able to
  fire; the index and stagger conventions, including that the record's
  `ua_new` is written after `lbc_lnk` and immediately before the swap, so
  the derived entering velocity is exactly the `un_e` read at line 535; the
  citation gate and its plant; and the empty production diff.

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

1. **Close or clear the Coriolis trend, and settle the disagreement this
   receipt records.** The record now carries NEMO's `cor_u`/`cor_v` at
   every substep. Substitute them, one variable, and measure the
   end-of-window velocity: if it goes AT-BAR the barotropic Coriolis
   operator is the statement and the fix is cited there; if it does not,
   the growth is a property of the coupled recurrence and the next operand
   to substitute is the surface-pressure gradient. This needs a per-substep
   override inside the compiled scan, which does not exist yet — it is the
   round's one new instrument. The discriminating quantity is already
   named: the per-cell response is a flat `2 … 3` times `f` while the
   response at the field maxima is 80 times the frozen operator's norm
   bound, and only the substitution tells which of those decides the
   window.
2. Re-test round 194's held candidate once the barotropic output is
   AT-BAR, under the unchanged gates (Decision 82).
3. The flux card's (`VORTEX-zco`) own kt=2 owner is untouched since round
   4; Decision 74's 30/15/10-km ladder stays blocked behind the vector
   kt=2 rows.
