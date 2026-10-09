# ORCA2 round 198 — pressure-gradient and explicit-drag hold

Date: 2026-10-09. Incoming tip: `f8c5a1b57`. Preregistration:
`9554dc1e9`. Measurement tip: `3f77991c1`.

## Result

**HELD.** Every number is **independent hierarchy rung 0**. The card starts
from its corrected climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry or rung-10 number is mixed into this result. The shipped
ORCA2 card, sea ice, all six ice selectors and its `unmeasured_features` tuple
are unchanged. `git diff f8c5a1b57..HEAD -- packages` is empty.

The complete substep-2 back-SSH calculation is bit-exact. All four
coefficients, the after/current/two older SSH operands, every weighted term,
both partial sums and the completed `sshp2_bck` are 0 / 26,640 unequal.
Prediction R198-P2 is therefore **REFUTED**: there is no non-bit back-SSH
input. NEMO computes that interpolation at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:642-650`, within the gated
span `ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:642-660`.

Given that exact recorded SSH, the production pressure statement still differs
on 1,226 cells, maximum `5.879156869962787e-07 m s-2`. Two statement-level
associations account for it. The candidate applies its extra compact V mask:
keeping the correct T-pivot north neighbour but applying that mask leaves 1,117
unequal cells at the same maximum. The candidate also uses a zero wall north
neighbour instead of the T-pivot fold association: with no extra mask, that
leaves 69 unequal fold-row cells, maximum `4.911895935325045e-07 m s-2`.
Using NEMO's fold association with no extra mask closes all 26,640 cells
bit-for-bit. The 1,117 and 69 diagnoses are controls, not an additive partition
of the 1,226 production differences. NEMO's compiled V statement is simply
`-zldg * (north - current) * r1_e2v`, with no `ssvmask`, at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:652-660` (gated by the
same `:642-660` span). Prediction R198-P3 is **REFUTED**.

The EEN half is bit-exact on all 15,875 active V faces: midpoint U and all four
V coefficients are exact, and the completed active-face Coriolis output is
0 unequal. The full array retains the registered 68 inactive northern-fold
cells, maximum `4.066345878944245e-08 m s-2`; this receipt does not relabel
those cells exact. NEMO calls and records the EEN result at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:663-670`.

The first active trend debt is the explicit bottom-drag coefficient. Candidate
`zCdU_v` is zero while NEMO records `-0.0004`: all 15,875 active faces differ,
with RMS and maximum absolute `0.0004`. Its two other operands, `vn_e` and
`hvr_e`, are bit-exact. The candidate drag replay differs from its passive
trace only by 6,599 signed zeros and zero magnitude. Replaying NEMO's recorded
Coriolis plus the compiled drag product closes completed `zv_trd` bit-for-bit.
The executed drag statement is cited at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:680-702`.

The pressure and trend remain a cancelling unit at the vector update. Pressure
alone leaves 15,943 unequal cells, maximum `9.009781378703638e-07 m s-1`;
trend alone leaves 563 signed-zero differences and zero maximum; both together
close 0 / 26,640 cells. Their consumer is
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:715-727`. No half is landed
and no trajectory census is claimed.

## Frozen source-ordered table

| boundary | unequal | maximum absolute | disposition |
|---|---:|---:|---|
| all back-SSH inputs and completed `sshp2_bck` | 0 / 26,640 | 0.0 | exact; R198-P2 refuted |
| production `zv_spg` | 1,226 / 26,640 | 5.879156869962787e-07 m/s2 | non-bit statement |
| fold-aware, no-mask `zv_spg` | 0 / 26,640 | 0.0 | exact NEMO statement |
| EEN `cor_v`, active V faces | 0 / 15,875 | 0.0 | exact |
| EEN `cor_v`, full array | 68 / 26,640 | 4.066345878944245e-08 m/s2 | registered inactive fold cells |
| drag `vn_e` / `hvr_e`, active | 0 / 15,875 each | 0.0 | exact |
| drag coefficient `zCdU_v`, active | 15,875 / 15,875 | 4.0e-04 | first active trend input |
| completed candidate `zv_trd` | 26,640 / 26,640 | 4.064540562696444e-08 m/s2 | non-bit |
| recorded Coriolis + source drag replay | 0 / 26,640 | 0.0 | exact |
| pressure-only vector substitution | 15,943 / 26,640 | 9.009781378703638e-07 m/s | not exact |
| trend-only vector substitution | 563 / 26,640 | 0.0 | signed zeros remain |
| pressure + trend substitution | 0 / 26,640 | 0.0 | exact |

The gate parsed both rank records exactly once across 65 substeps and required
every stream. The passive traced and untraced states remain array-identical in
SSH, U, V, both external modes and both transport averages. The derived-drag
subtraction control is non-vacuous: `trd_v - cor_v` differs from the direct
source replay on 25,692 cells, maximum `1.3002245253567904e-23`, including
10,585 signed-zero differences.

The measurement artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round198/pressure_trend_split.json`
(SHA256 `5ccc36cf288b7e2a063f8ab8867829815560b4047718f015e28a40a0eba77b71`).

## Prediction ledger

| prediction | verdict |
|---|---|
| R198-P1 record sufficient | **CONFIRMED**: both ranks and all required operands admit; no acquisition. |
| R198-P2 back-SSH carries first debt | **REFUTED**: all inputs, terms, partial sums and output are bit-exact. |
| R198-P3 recorded SSH closes production V gradient | **REFUTED**: mask/fold association leaves 1,226 cells until both match NEMO. |
| R198-P4 EEN exact active; drag first active debt | **CONFIRMED**: active EEN exact; `zCdU_v` is the first active trend difference. |
| R198-P5 atomic pair closes; singles do not | **CONFIRMED**: pair 0 unequal; both single substitutions remain non-exact. |
| R198-P6 controls bind | **CONFIRMED**: all four plants refuse with exit 2. |

## Validation and choices

The source-registry, coefficient-bit, drag-identity and missing-stream plants
all refuse with exit 2 (plant log SHA256
`ba5dc3ecc60c5ea0416ce5c8c840d3db1a3c1d1fab519f6e084536dc67961cfa`).

Independent review unavailable in-sandbox: the required separate
`codex exec --sandbox read-only` invocation failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system
(os error 30)` (log SHA256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).
No independent verdict is claimed.

The default citation audit passes 274 citations with zero failures, unmapped
citations or failing map entries (SHA256
`df25db4dadb79d12754e58bde924b18298bdf0a870d2257617772e7886da1c9c`).
This receipt passes six citations with the same zero counts (SHA256
`64c9a39c1a815d876bd7a64605ba921dfb44e494c54ebc85e3447d6af838ce41`).
Shifting the `:642-660` span by two lines fails as required with
`SYMBOL-NOT-AT-LINE` and exit 1 (SHA256
`f75721e90cf5037e3d2515eaaf26fb7773fc1efa10da1e916f0f70f9094e787c`).

The final focused set (rounds 146 and 195-198 plus the citation gate) passes
68/68 in 4.36 s (JUnit SHA256
`97c2c3d1176177f8a6f3eaa9f466997d7444c14763f565afb6ee68a370246d18`).

The required single `tests/ocean/fidelity -n 12` battery collected 2,952
tests and reached 97%. It displayed two registered reds before it stopped
producing progress: the worktree-stamp ratchet and the GYRE spread-floor
record gate. After several silent minutes it was interrupted once; it emitted
no terminal XML and was not rerun. This is an incomplete battery, not a green
claim (log SHA256
`be9a0c98db2db4595e0eb70e9f12599fd1b0a31197290c001bee3ab766c2050b`).
Both displayed IDs were rerun alone and reproduced: 13 report emitters remain
unstamped, and the certified GYRE year harness moved after the registered
members ran (JUnit SHA256
`3924ae7ae58b715fe8fb1500670b804a1fabd0e748cf57b973bbd566a24f7bc2`).
Neither failure touches the round-198 instrument; its six tests and the
citation-map audit passed within the full battery.

No configuration choice, carried-state change, stabiliser, tolerance, NEMO
source change or executable observer was introduced. ASKED choices: round
197's OPEN pressure-gradient and completed-trend split. UNASKED choices: empty.

## OPEN

Round 199 builds one private atomic unit: (1) NEMO's V pressure statement with
the T-pivot north association and no extra compact V mask, plus (2) NEMO's
explicit substep drag coefficient `-0.0004`. Prove both statement outputs
bit-exact side by side, preserve and register the known 68 inactive EEN fold
cells, then re-score the complete fold/transport arm under Decision 96 and all
trajectory gates. Do not partially land either statement.
