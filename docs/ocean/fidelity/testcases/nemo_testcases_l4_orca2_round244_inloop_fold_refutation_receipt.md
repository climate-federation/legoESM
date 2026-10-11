# ORCA2 round 244 — in-loop fold association refutation

Date: 2026-10-11. Frozen base: `24152301e`. Preregistration commit:
`4bb0875e6`. Measurement commit: `923e09e13`. Status: **HELD**. Evidence
root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round244/`.

This round changes no effective package/model file, deck, selector, carried
state, stabiliser, sea-ice setting, or `unmeasured_features` entry. Two
source-exact candidate commits were measured and then reverted; the effective
package tree is identical to measurement commit `923e09e13`. Results below
are separately labelled **independent** and **given NEMO's entry**. The two
labels produce identical counts and maxima.

## Compiled statement and protocol

The executing OMT-4 build calls one in-loop northern-boundary association for
`ua_e`, `va_e`, `hu_e`, `hv_e`, `hur_e`, `hvr_e`, and `ssha_e` after every
external substep, with U/V velocity sign -1 and scalar depths/SSH sign +1, at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-739`.

The committed round-244 gate reuses the admitted round-222 OMT-4 twins and
the passive production trace, scores substeps 1..6, and refuses a candidate
unless the midpoint V velocity, live V depth, and their metric transport are
all exact. Execution is production JIT on CPU with fp64, x64, and libm. The
ordinary and observed completed states are array-identical. Both twins are
exact on every defined payload. Label-coverage, substep-order, fold-sign,
passivity, pair-census, and one-bit plants each exit red.

## Frozen predictions

| prediction | result |
|---|---|
| R244-P1: admitted, passive measurement agrees under both labels | **CONFIRMED** |
| R244-P2: recorded NEMO boundary obeys the compiled association | **CONFIRMED** on defined record support |
| R244-P3: first candidate fold violation names the missing boundary | **CONFIRMED**: SSH at substep 3, 35 cells |
| R244-P4: a complete association moves/closes the 129/120 pair | **REFUTED**: the exact counts remain 129/120 |
| R244-P5: one complete statement is eligible to land | **REFUTED** by the pair gate |
| R244-P6: all controls fire | **CONFIRMED** |

## Baseline fold and pair table — both labels

Substeps 1 and 2 have zero candidate fold-identity violations in SSH, U, and
V. The first violation is the substep-3 exit SSH: 35 unequal cells, maximum
7.090580641730404e-09 m at fold index 45. U and V remain fold-identical
through substep 6. The registered cancelling pair evolves as follows:

| substep | midpoint V unequal / max | live V depth unequal / max | `zhV` unequal / max |
|---:|---:|---:|---:|
| 3 | 0 / 0.0 | 35 / 0.0033989498219852976 m | 35 / 0.1377211210783571 m3/s |
| 4 | 66 / 1.065982855371489e-10 m/s | 68 / 0.008802612408089772 m | 68 / 0.5600948002538644 m3/s |
| 5 | 100 / 5.15086013772037e-10 m/s | 99 / 0.014197753081134579 m | 100 / 1.3417117258068174 m3/s |
| 6 | 129 / 1.5051524978638986e-09 m/s | 120 / 0.019759179072480038 m | 129 / 2.690599228371866 m3/s |

The first candidate attempt associated all seven values inside the helper but
did not carry its returned SSH into the next substep. It moved no measured
row and is retained as `helper_only_REFUTED_*` evidence. The second attempt
also carried the associated SSH. It closes every candidate fold identity at
all six substeps, so it is a faithful test of the proposed boundary.

That complete candidate nevertheless leaves the exact substep-6 censuses at
129 midpoint-V, 120 live-depth, and 129 `zhV` cells. The maxima become
1.0686990884259395e-09 m/s, 0.019759179072480038 m, and
2.3249768203822896 m3/s respectively. The fail-closed classifier refuses:
`independent: candidate substep-6 v_mid is not exact`. Therefore the missing
association is real but is **not** the owner of round 243's pair. Both
candidate changes were reverted; no partial association lands.

## Evidence, tests, and review

The baseline reports hash to `97b4c21d...` (independent), `c542dc1a...`
(given NEMO's entry), and `ae5c5d67...` (classification). Candidate reports
hash to `37977af0...` and `bfc6d454...`; the refusing classifier log hashes
to `4b4bb76f...`. Full SHA256 values are retained in the round evidence.

Focused tests cover measurement schema, all six plants, both-label
classification, and candidate refusal. The citation gate audits this receipt
and the campaign default receipt; its planted shift must fire. The required
once-only fidelity battery result is recorded below after completion.

`codex exec --sandbox read-only` failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system
(os error 30)`. **Independent review unavailable in-sandbox.** The exact
output is retained as `independent_review.log`; no verdict is invented.

## OPEN

Return to the source-ordered pair named in round 243. Split the substep-6 AB3
midpoint-V producer across its current, previous, and previous-previous V
histories, and split the midpoint-depth producer across raw `hv_0`, midpoint
SSH, area interpolation, and `ssvmask`. Use the current production D114 mesh
route and score the two statements atomically against `zhV`; no partial
operand lands. The admitted exit histories reconstruct the AB3 inputs and the
existing trace carries the depth inputs, so no new NEMO acquisition or user
decision is needed.
