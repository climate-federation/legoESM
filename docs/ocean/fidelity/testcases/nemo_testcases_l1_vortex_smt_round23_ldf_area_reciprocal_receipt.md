# Receipt — VORTEX_SMT round 23 (lane round 235): LDF stored area reciprocal and final seam

**Status: HELD.** No physics, card, carried state, trajectory, or certified
number changed. NEMO's recorded `r1_e1e2t` and the locally source-rounded
stored reciprocal are each bit-exact in the production step, but neither moves
the remaining `8,613`-cell / `2.5978639524806749e-14 K s-1` LDF RHS residual.
Removing the final wet mask is also inert. Reproducing NEMO's in-place `Krhs`
add and the recorder's subtraction reduces the unequal-cell count to `2,551`
but leaves the full maximum, so it is not the magnitude owner. The
preregistered closure predictions are refuted and no trajectory landing gate
is eligible.

Base: `12b192dbc` (round 234). Preregistration: `010bd21cf`. Stored-area
instrument: `db2ad2aea`; bounded compilation: `d3cf7fba8`; final-seam
instrument: `256fd59ab`; citation map: `85fef7d70`. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round235/`.

## 1. Compiled statements and controls

The exact compiled build stores the area product and then its reciprocal at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90:155`. The regular
LDF update consumes that stored reciprocal in its in-place `Krhs` update at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`; the
deepest-level form is at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.
The admitted instrument records `rhs_before`, `rhs_after`, and their difference
at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/vortex_r23_ldf_terms.f90:149-151`.
These are the only source statements claimed in this round.

The production-JIT walk first reproduced the round-234 control: exact literal
horizontal and vertical fluxes leave `8,613` unequal wet cells at
`2.5978639524806749e-14 K s-1`, with tendency digest
`6300f14abb814b3af465a3439a143f9c8ff2d2f8ae38d1f76de39787a108d542`.
All hmsku/hmskv, A11/A22/A13/A23, dit/djt/dkt, both horizontal fluxes and both
vertical-flux controls remain bit-exact.

| production-JIT arm | reciprocal unequal | RHS unequal cells | RHS max abs (`K s-1`) | result |
|---|---:|---:|---:|---|
| round-234 exact-divergence control | n/a | `8,613` | `2.5978639524806749e-14` | reproduced |
| recorded `r1_e1e2t` | `0` | `8,613` | `2.5978639524806749e-14` | inert |
| local stored product then reciprocal | `0` | `8,613` | `2.5978639524806749e-14` | inert |
| recorded reciprocal, no final wet mask | `0` | `8,613` | `2.5978639524806749e-14` | inert |
| recorded reciprocal, NEMO `Krhs` add/sub | `0` | `2,551` | `2.5978639511571860e-14` | count only; magnitude unchanged |
| recorded fluxes and operands, post-hoc floor | `0` | `8,451` | `2.9778502051908996e-22` | comparison floor only |

The stored-area report is
`area_reciprocal_walk_retry.json` (SHA-256
`92a4475693ec61ab3d8d36d4e7a31f48505e48262f73f1b8c01bd176fb78353d`);
the final-seam report is `final_seam_walk.json` (SHA-256
`95b5a18126aad2a3eb3bd07d53fdf0105d026d2dec81ddc829defa686e49db88`).

The first attempted measurement compiled every historical arm and exhausted
host memory. Its log contains `LLVM ERROR: Unable to allocate section memory!`.
Because that command omitted `pipefail`, the shell pipeline incorrectly
returned zero. It is explicitly **INVALID evidence**. Commit `d3cf7fba8`
bounded each measurement to its frozen control and two discriminating arms;
all accepted measurement and plant commands used `set -o pipefail`.

## 2. Predictions, refutations, and boundary verdict

- R23-P1 is confirmed: the exact-divergence control reproduced all frozen
  values.
- R23-P2 is **REFUTED**: substituting the recorded stored reciprocal does not
  approach the comparison floor.
- R23-P3 is **REFUTED**: the locally source-rounded product/reciprocal is
  bit-exact, but the RHS digest and residual are unchanged.
- R23-P4 is **REFUTED as a magnitude closure**. The final mask is inert; the
  `Krhs` seam removes `6,062` unequal cells but changes the maximum by only
  `1.3234889e-23 K s-1`, leaving the whole `2.60e-14 K s-1` magnitude.

The first non-bit source statement after exact fluxes therefore remains the
regular/deepest RHS update at lines 306-310/327-331, but the stored reciprocal,
wet mask, and in-place accumulation/subtraction are exonerated as magnitude
owners. No narrower source owner is claimed. This is the last measured
same-stage boundary: further source-rounding splits would be a bit walk below
the campaign's magnitude priority.

R23-P5 is confirmed. A one-ULP recorded reciprocal change makes its operand
row non-bit and prints `STATUS PLANT-FIRED`; a one-ULP `rhs_before` change
moves the final-arm unequal count from `2,551` to `2,552` and also prints
`STATUS PLANT-FIRED`. Both plant commands exit `1`. Their report SHA-256 values
are `f0e6bfac20cddcc53a4e21fc2e812522fe089ec810b4428108641c34ca08eb8c`
and `c58d08b46b8f5761096284b0c4b44b7a801b6c2106f740650640fe1ecb342020`.

## 3. Landing and blast radius

No candidate closes locally, so the ladder/year landing gate is not entered.
Every new selector is reachable only through the private write-only diagnostic
hook; the production default and all card configurations are unchanged. The
certified GYRE values therefore remain:

| row | certified value/status |
|---|---:|
| kt2 T / S | `6.054357687161262e-16 / 5.786374251651969e-16`, AT-BAR |
| kt2 U / V | `8.326672684688674e-17 / 9.71445146547012e-17`, AT-BAR |
| kt3 T / S | `5.861944241472192e-10 / 1.6244926507906096e-11`, DEBT |
| day 30 T rms | `2.3432437414839976e-06 K` |
| day 240 T rms | `6.5817049818294640e-05 K` |
| day 360 T rms | `5.4077372201617810e-05 K` |

DINO shares the production operator, so the mandatory month integration ran
despite the private-only change. The model completed 960 steps. Its first
scoring attempt correctly refused the dirty pending citation-map edit; the
already-produced snapshot was then scored from the clean committed tree in a
fresh directory. The mechanical verdict is:

> DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.053801168e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS

LOCK_EXCHANGE, OVERFLOW, the generic GYRE recipe, both SMT cards, all six flat
VORTEX cards, the certified GYRE ladder/year and ORCA2 cannot select the
private modes and are not re-baselined. For ORCA2, the stored reciprocal and
final seams are **UNMEASURED** on its own developed state but share the cited
statements; this round exonerates them on the SMT-3 developed-state record.

**UNASKED list: EMPTY.** No physics, configuration, default, threshold,
record source, state, or stabiliser choice was made.

## 4. Tests, citations, and review

The direct source-boundary unit control reports `1 passed in 8.97s`. The final
focused suite reports:

> 138 passed in 182.55s (0:03:02)

The round citation gate finds four mapped citations, zero failures, zero
unmapped citations and zero map-audit failures. The cumulative default receipt
gate passes with 274 citations and the same three zero counts. Adding the four
private hook lines shifted historical citations in
`ocean_model_latlon_cgrid.py`; both the map and the default receipt were
mechanically re-anchored by +4 below the insertion, while the one spanning
range grew from 440 to 444 lines. Shifting the round's domhgr.f90 line 155
citation by two lines makes the gate print `SYMBOL-NOT-AT-LINE` and exit `1`.

The required read-only Codex review was attempted after the diff and evidence
were complete. Its complete verdict transcript is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**; there is no reviewer
`DO NOT SHIP` verdict. No physics candidate lands.

## 5. OPEN

Round 236 stops this residual bit walk and lands the already measured SMT-3
partial-cell mask-plus-live-divisor pair under the full trajectory/card gates,
registering every moved row. It must re-prove the pair at the current tip,
preserve the exact compiled citations, measure GYRE ladder/year, both SMT
registries, six flat VORTEX registries, tanks, generic GYRE, the private DINO
month gate, and write the precise ORCA2 merge pointer. A red ratchet holds the
candidate. After that landing, proceed to SMT-4 lateral momentum diffusion;
do not spend another round splitting the `2.60e-14 K s-1` same-stage residue.

No NEMO acquisition and no configuration decision is required.
