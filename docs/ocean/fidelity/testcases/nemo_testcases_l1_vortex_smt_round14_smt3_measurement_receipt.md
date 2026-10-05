# Receipt — VORTEX_SMT round 14 (lane round 226): SMT-3 tracer diffusion

**Status: LANDED as a measurement rung; the internal tracer-LDF statement is
HELD.**  The explicit `VORTEX_SMT3_VEC-zps` card and its reusable gates land,
but no arithmetic change is claimed as a fidelity fix.  The admitted record
localises the kt=2 temperature magnitude to the compiled `tra_ldf` call; it
does not carry the slope/flux intermediates needed to name a statement inside
`ldf_slp` or `traldf_iso_lap`.  That distinction is kept open rather than
inferred.

Base: `be6e4fb91` (round 225).  Preregistration: `a07d64049`.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round226/`; immutable NEMO
record: `phase3/round224/oracle_vortex_smt3/`.

## 1. Record admission and resolved experiment

The operator-run wrapper ended with all three ready lines.  Both admission
JSON files say `ADMITTED`; the 10-step and 100-day reference/instrumented
restarts are byte-identical, the 27 and 3,067 self-describing records parse to
EOF, and the header/name/truncation plants exit nonzero.  Both NEMO runs report
`STOP 0`.  No hand-predicted byte count or header tuple is used.

The new build's own resolved output selects this one-module delta from SMT-2:

| resolved `namtra_ldf` value | SMT-2 | SMT-3 |
|---|---:|---:|
| `ln_traldf_OFF` | T | F |
| `ln_traldf_lap / iso / msc` | F/F/F | T/T/T |
| `nn_aht_ijk_t` | reference, inactive | 20 |
| `rn_Ud / rn_Ld` | reference, inactive | 0.018 m/s / 200,000 m |
| `rn_slpmax` | reference | 0.01 |
| `blp / lev / hor / triad / triad_iso / botmix_triad` | F | F |

The active six values are ORCA2 rung 0's own deck values.  The companion
values are the reference namelist values that deck leaves unset; every one is
validated by the card, so no library default chooses a physical branch.

The compiled SMT-3 branch reads the reference and configuration namelists in
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:231-233`, resolves
laplacian standard-isoneutral diffusion at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:261-283`, and mode
20 calls `ldf_c2d` with `0.5*rn_Ud` at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:354-390`.  The
dispatcher sends that resolved `np_lap_i` to `traldf_iso_lap` at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90:105-110`.

## 2. Exact card and geometry

`VORTEX_SMT3_VEC-zps` retains SMT-2's S-EOS, partial-cell mesh, background
mixing/EVD, implicit linear drag, vector-EEN momentum, external mode, timestep
and 100-day length.  It adds only the explicit GM/Redi configuration that
transcribes the table above.  Two plumbing gaps had to be closed for that
configuration to execute NEMO's selected program:

1. every native-slope density/EOS call now receives the card's explicit
   S-EOS coefficients instead of an implicit module fallback;
2. WS-RK3 constructs the carried step-entry N2 bundle whenever native slopes
   request it, independently of whether the card also selects a TKE closure.

That second condition follows NEMO's program order, not a configuration
choice: `VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:159-177`
calls `eos_rab`, `bn2`, `zdf_phy`, `eos`, and `ldf_slp` before entering the
three stages.  It therefore executes on this
constant-background-mixing card even though no TKE closure is selected.

The extended existing geometry gate compares 18 rows.  All are exact: k_bot,
reference e3t/e3u/e3v/e3f, the four card operands, resolved qco U/V faces,
northern/eastern boundary handling, e3w, and T/U/V masks.  Its non-vacuity
row sees 1,164 cells where the correct e3u differs from e3t, so an aliased
face field would fail.  Verdict: `GEOMETRY IDENTICAL`.

## 3. Preregistered predictions

| prediction | result | verdict |
|---|---|---|
| R14-P1 record admission | both arms admitted; restart identity and all plants pass/fail as designed | CONFIRMED |
| R14-P2 one-module explicit card | configuration diff is `namtra_ldf`; 18 geometry rows are exact | CONFIRMED |
| R14-P3 ladder order | all five kt=1 rows AT-BAR; first over remains kt=2 | CONFIRMED |
| R14-P4 pre-LDF at the bar, post-LDF non-bit | pre-LDF T is already DEBT at 7.418332614861356e-11 K; post-LDF is 2.0915088416728622e-07 K | **REFUTED** |
| R14-P5 shipped-length sanity | 3,000 steps complete; every daily T/S/u/v/ssh row is finite; kt=1..10 sanity reproduces the ladder | CONFIRMED |
| R14-P6 no premature arithmetic landing | no internal LDF statement is named or changed | CONFIRMED |

R14-P4 is not softened.  The pre-LDF discrepancy is real.  The magnitude
jump across `tra_ldf` is also real, but a non-bit input plus a non-bit output
does not identify which internal statement owns it.

## 4. Certified kt=1..10 registry

The complete 50-row registry is `round226/smt3_ladder.json`.  Representative
rows are:

| kt | T | S | u | v | ssh |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 2.220446049250313e-16 | 2.220446049250313e-16 | 1.3552527156068805e-20 |
| 2 | 1.020298116571876e-08 | 6.090366306515142e-16 | 3.1789753815458788e-09 | 6.728428985844359e-10 | 3.938355197519172e-10 |
| 10 | 1.9599804535277618e-07 | 1.8271098919545404e-15 | 8.1959957819503e-07 | 3.2900322209729707e-07 | 3.0496309144645295e-08 |

All kt=1 fields are AT-BAR.  The first-over-bar row is kt=2, unchanged from
SMT-2.  Salinity remains AT-BAR at kt=2.  This is a new rung, not a candidate
before/after arm, so no moved-row direction is invented; all 50 absolute rows
are registered.

## 5. Stage-3 magnitude boundary and the exact open claim

The compiled stage calls tracer advection and surface forcing before the
stage-3 lateral diffusion, then vertical diffusion:
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:456-488` and
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:501-546`.  The
stage-3 `tra_ldf` call itself is
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:532`;
`tra_zdf` follows at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:546`.

The production-JIT walk, driven from NEMO's recorded stage entry, reports:

| boundary | unequal T cells | max abs T | relative |
|---|---:|---:|---:|
| NEMO stage-entry tracer | 1,691 | 3.552713678800501e-15 | 1.732931465374028e-16 (AT-BAR) |
| pre-LDF RHS | 14,220 | 7.418332614861356e-11 | 3.6187800385900646e-12 |
| post-LDF RHS | 15,948 | 2.0915088416728622e-07 | 1.0202982479596311e-08 |
| completed T | 14,173 | 2.0915085485739837e-07 | 1.020298116571876e-08 |

Substituting NEMO's recorded stage-3 transports leaves the completed error at
2.0915085485739837e-07 K.  Feeding legoESM's own transports back through the
same seam moves only nine T cells at 3.55e-15 K, controlling the hook.  Thus
the FCT transport discrepancy is bounded out as the magnitude owner, while
the LDF call introduces 2.090767008411376e-07 K of additional max error and
carries the completed kt=2 magnitude.

**First named compiled boundary:** the call from `tra_ldf` into
`traldf_iso_lap`, selected at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90:105-110`.  Inside
it, NEMO first computes A33, then tracer gradients, the horizontal tensor
factors/fluxes, and the vertical factors/divergence in
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:154-304`.  The current
record contains only the aggregate before/after RHS, not these intermediate
arrays.  Therefore this receipt does **not** claim that slope limiting, MSC,
the coefficient, a face thickness, or the final divergence is the first
non-bit statement.  That internal first statement is UNMEASURED.

## 6. Shipped 100-day trajectory

The NEMO and legoESM arms both run the shipped 3,000 steps of 2,880 s.  The
same scorer used for flat VORTEX reports:

| day | T rms | u rms | v rms | ssh rms |
|---:|---:|---:|---:|---:|
| 1 | 2.90533338091134e-07 | 7.086189758e-08 | 8.50807762e-08 | 1.933063117e-08 |
| 10 | 7.8867120887e-06 | — | — | — |
| 30 | 1.1373928051e-04 | — | — | — |
| 60 | 2.1593031319e-04 | — | — | — |
| 100 | 6.813785267886451e-04 | 1.4051094681706748e-04 | 1.3699338070109296e-04 | 7.726694673482761e-05 |

At day 100 the maxima are 1.8923906065542617e-02 K, 4.485077642930893e-03
m/s, 3.79836676398047e-03 m/s and 9.88776987414286e-04 m.  Every daily field
is finite.  The prior SMT-2 context is 4.352692968282214e-05 K at day 100;
the rung-to-rung difference is registered but is not a one-variable physics
candidate because SMT-2 also contains the preceding EVD and bottom-drag
modules.

## 7. Shared-card blast radius

The certified GYRE ladder has 954 rows and no headline value or status moves;
first-over-bar remains kt=3.  The full 360-day arm reproduces the current
round-223 certified T rms values exactly at every scored day:

| day | T rms (K) |
|---:|---:|
| 30 | 2.3432437414839976e-06 |
| 60 | 1.4793243459436834e-05 |
| 90 | 1.6332701526871403e-05 |
| 120 | 1.0965908847407414e-04 |
| 180 | 6.1153356819063490e-05 |
| 240 | 6.5817049818294640e-05 |
| 300 | 5.4660485988812500e-05 |
| 360 | 5.4077372201617810e-05 |

The first scoring attempt used the 30-day `year_owners` root and correctly
refused its missing day-60 restart.  The reported table is the corrected run
against `phase3/year_fromrest`; the refusal is retained in the log.

The DINO month gate, using a private work directory, reports:

```text
DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.053801168e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS
```

The existing-card registry census and final test summaries are recorded in
section 9.

## 8. ORCA2 pointer

ORCA2 rung 0 executes this same compiled branch: laplacian standard
isoneutral diffusion with MSC and coefficient mode 20.  Its exact next walk
is the same ordering recorded here: `ldf_slp` outputs first, then the
`traldf_iso_lap` A33, gradient, horizontal-flux, vertical-flux and divergence
boundaries.  The VORTEX result does not establish which one is wrong on ORCA2;
it establishes the record fields the ORCA2 lane must compare.  No ORCA2
trajectory is claimed by this round.

## 9. Tests, citations, review, and controls

The focused CPU/fp64 suite reported:

```text
179 passed in 237.89s (0:03:57)
```

The recipe-derived regression census reran SMT-2, SMT-1, the two base SMT
cards, all six flat VORTEX cards, LOCK_EXCHANGE and OVERFLOW.  Each of the
twelve 50-row cellwise comparisons is `PASS`: zero improved cells, zero
worsened cells, zero row-status changes, zero first-over-bar changes and
maximum oracle-residual worsening `0.0` ULP.  The JSON comparisons are under
`round226/inert/compare_*.json`; this is exact invariance, not merely a bar
classification.

FINAL_BATTERY_PLACEHOLDER

The real receipt citation gate reports 10 citations, zero unmapped citations,
zero failures and zero audit failures.  The shifted-citation plant moves the
`ldftra.f90:354-390` span by one line, exits nonzero and reports a source-anchor
failure.  Because this round edited already-cited Python files, the historical
default receipt and citation map were re-anchored together by the repository's
sequence-mapped re-anchor tool; the default-receipt audit is clean.

FINAL_REVIEW_PLACEHOLDER

## 10. Choices and OPEN

**UNASKED list: EMPTY.**  Decision 93 already authorises this rung.  No new
scheme, coefficient, stabiliser, threshold, carried state or default is
selected.  The card states every consumed option.

OPEN, in order:

1. Extend the existing SMT tracer record—not a second harness—with
   self-describing groups for `ldf_slp` outputs and the compiled
   `traldf_iso_lap` A33/gradient/horizontal-flux/vertical-flux/divergence
   boundaries.  The current record cannot name their first non-bit statement.
2. Walk those boundaries under production JIT given NEMO's stage entry.  A
   statement may land only with a one-variable local proof and the full SMT,
   flat-VORTEX, tanks, GYRE ladder/year, generic-card and DINO-month gates.
3. After SMT-3 closes or is explicitly held, continue Decision 93 with SMT-4
   lateral momentum diffusion; do not infer its partial-cell operand from the
   tracer result.
