# SWG double-gyre round 4 receipt

Date: 2026-10-12

Status: **STOPPED FOR DECISION — THE NEW-DECK EXACT-BIT PREDICTION IS
REFUTED**.  Both admitted ladders were run through the production JIT on CPU
under the certified fp64/libm policy.  The earliest difference is timestep 2,
stage 1; the 30-day trajectory differs from day 1.  Both one-ULP controls bind.
No physics or numerical method was changed, and no NEMO build, NEMO run, or GPU
run was launched in this round.

## Provenance and scope

The incoming tip was `402ba0c4b9bcf2bde0c384994bd4c45ae1f834de`.
Preregistration is commit `037a73248`; ladder and climate-instrument
implementation is commit `791b2810fe52576734a6c44ecfd73ff1f59acea9`;
and the independent-review corrections are commit
`a028d0c51af2d5383588512c03cd6324dadc3ea4`.
The admitted stage and month records remain content-pinned by their acquisition
admissions.  Raw stage frames are 648 by 648 at 6.25 km (43.67 MB each) and 168
by 168 at 25 km (2.94 MB each); the recorded production bounds supplied to the
card are respectively 644 by 644 and 164 by 164, containing 640 by 640 and 160
by 160 wet points.

There is no change under `packages/`, no physics change, and no new numerical
scheme.  The no-argument shipped card remains the default.  Its digest
regression remains `f9d46bfd08f23702`, and its previously certified
given-entry 338/338, chained 338/338, independent 50/50, carried 29/29, and
month 690/690 rows are unchanged.  The current-tip card and mounted shipped
regressions are part of the archived direct-test battery.

## Ten-step honesty ladder

The canonical every-row reports are `round4/stage_given.json` and
`round4/stage_chained.json` under the external evidence root.  Both carry the
clean implementation SHA.  The requested exact-bit gate classifies every
non-identical row as DEBT even when the separate normalized `1e-15` numerical
bar passes.

| Arm | Rows | AT-BAR | DEBT | First DEBT | Largest absolute residual |
|---|---:|---:|---:|---|---:|
| given entry | 338 | 276 | 62 | kt 2, stage 1, SSH | `1.0842021724855044e-19` m, kt 10 stage 1 SSH |
| chained | 338 | 113 | 225 | kt 2, stage 1, SSH | `2.168404344971009e-19` m, kt 10 stage 1 SSH |

The first unequal given-entry row is `given.kt2.s1.ssh_k`: 1,278 unequal wet
cells and maximum absolute residual `3.3881317890172014e-21` m.  Given-entry
stages 2 and 3 are completely bit-identical (117/117 and 104/104 rows).  Thus
the measured first-divergence boundary is the whole-step-to-stage-1 transition;
no owner is assigned in this round.  Chaining carries the last-bit difference
into later stages, as expected after the given-entry result.

The preregistered stage plant advances the nonzero admitted `u_k[481,2]` by
exactly one ULP.  Production SSH changes at two wet cells with maximum absolute
change `1.3234889800848443e-23` m.  Its complete result is
`round4/stage_ulp_plant.json`.

## Thirty-day honesty ladder

The canonical `round4/month.json` reports all 690 preregistered rows: 249
AT-BAR and 441 exact-bit DEBT.  The first difference is day-1 entry
`vtau_before`: NEMO and legoESM have opposite zero signs at 160 wet cells and
therefore zero numerical residual but different IEEE words.  The same signed-
zero difference appears in `vtau_before`, `vtau_now`, and restart `vtau_b` on
all 30 days, accounting for 90 rows.  The first nonzero difference is day-1
restart SSH at kt 432: 22,487 unequal wet cells and maximum absolute residual
`9.367506770274758e-17` m.  The first row beyond the separate normalized
`1e-15` numerical bar is day-7 restart SSH at
`1.2281842209915794e-15` m.  The largest absolute residual at day 30 is
`1.6431300764452317e-14` m/s in `un` (normalized residual
`9.20175294902847e-15`).  The report contains 390 daily-entry rows (69 AT-BAR,
321 DEBT) and 300 restart rows (180 AT-BAR, 120 DEBT).

For the month plant, all ten fields reconstructed from the admitted day-1
restart are bit-identical before perturbation.  Advancing the largest-magnitude
nonzero wet `u[160,37]` by one ULP changes production SSH at two wet cells by
at most `1.3877787807814457e-17` m.  The complete control is
`round4/month_ulp_plant.json`.  `round4/ladder_summary.log` is a compact
machine-derived index; the JSON files remain authoritative and report every
row.

These results refute the preregistered claim that the new deck would be
bit-identical at every row.  They do not authorize a physics change: the next
fidelity action is to localise the signed-zero wind path and the independent
kt-2 stage-1 transition before considering any correction.

## Climate-comparison instrument and year-1 refusal

The committed streaming instrument assembles overlapping per-rank IOIPSL tiles
by their T/U/V coordinates and refuses missing or conflicting coverage.  It
pins the NEMO executable and resolved deck and checks the legoESM manifest's
physical namelist.  An explicit start day plus length selects year 1 or days
366-1095 for the registered years 2-3 window.  For identically sampled windows
it computes time-mean SSH, U, V, and speed; an EKE map; basin-area-weighted KE
and EKE time series; the mean eastward jet-core latitude and maximum-eastward-
speed profile at every longitude; and the shared isotropic KE spectrum.  It
writes content-digested arrays, a JSON summary with per-metric right-minus-left
bias, mean absolute difference, RMS difference, and maximum absolute
difference, and a common-scale side-by-side figure.

Seven direct synthetic tests cover a known analytic mean/EKE/KE/jet answer,
identical-input spectral equality and zero cross-model statistics,
coordinate-halo deduplication, missing and conflicting tile plants, direct
later-window selection in both readers, a wrong-deck plant, a shifted NEMO
clock plant, an explicit daily-mean-versus-snapshot refusal, and all three
output artifact kinds.  The year-1 run is registered as **year 1, spin-up**;
years 2-3 remain the real climate window.

No year-1 climate result is reported.  At the measurement point the legoESM
365-day non-certified GPU run was complete, while the active NEMO run's 32
T-rank files still exposed an empty `time_counter`; the committed command
therefore refused with exit 2 before creating an output directory, as archived
in `round4/climate_year1_attempt.log`.  There is also a protocol-level mismatch
to resolve before rerunning: the compiled NEMO card requests daily means,
whereas the existing legoESM files are end-of-day snapshots and their manifest
does not declare matched averaging.  The instrument refuses unequal sampling;
it will not manufacture a comparison from those files.

## Operator GPU profile folded forward

The operator's Round-3 GPU-1 manifest remains explicitly **non-certified**.
At 640 by 640 wet cells the Python loop measured `59.9095 ms/step` and the
compiled-day scan `4.49480 ms/step`, a `13.3286x` speedup that meets the
`<5 ms/step` target.  Two-day scan-versus-step maximum absolute differences
were `2.250803710079907e-16` m for SSH, `1.0061396160665481e-16` m/s for U,
and `1.4224732503009818e-16` m/s for V, all within the registered `1e-12`
tolerance.  Source evidence is
`round3/gpu_profile_2d_operator/manifest.json`; these are operator-run native
transcendental numbers, not certified ladder results.

## Tests, preflights, and evidence

The exact current-tip battery and terminal count are archived in
`round4/direct_tests.log`: **212/212 passed** on CPU with x64 enabled.  It
includes the card, mounted ladder, geometry, acquisition-record, GPU-driver,
and new climate-instrument tests.  `git diff --check` and focused Ruff checks
pass.

All four no-run acquisition preflights pass at the implementation commit:
shipped stage, shipped month, Maddison stage, and Maddison month.  Their four
logs under `round4/preflight_*.log` show that every writer target would still
be created from shipped `SWG`; the selected deck is overlaid afterward; and
the shipped and Maddison daily cadences remain 24 and 432 steps.  No
`makenemo`, `mpirun`, or GPU command was run.

The evidence root is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round4/`.
`evidence_sha256.txt` pins the structured ladder reports and direct-test log.

Both independent reviews initially returned HOLD.  Their confirmed findings
were dispositioned before final review: the exact gate now compares raw IEEE
words and has a signed-zero plant; the month control carries deck/acquisition
provenance; the climate instrument admits the deck, supports arbitrary
windows, emits cross-model scalars, and directly tests its clock and spectrum
controls.  All ladder evidence and the direct battery were regenerated at the
corrected commit; no pre-fix exact count is retained as evidence.

## Decision and acquisition boundary

Recommended next round: retain the exact-bit result as fidelity debt, localise
both the signed-zero wind path and kt-2 stage-1 transition without changing
physics, add a daily-mean output mode to the non-certified GPU driver, and
reacquire the legoESM year-1 window before running the climate instrument.  The
existing snapshot year must not be relabelled as a mean.  No immediately valid
acquisition command exists until that driver mode is implemented, so
acquisition is NONE pending this decision.

## Oracle citations

The compiled daily writer selects IOIPSL averaging and sets its output interval
to `nn_write * rn_Dt` at
`SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/diawri.f90:146-160`.  This is the executed
output semantics that make the existing end-of-day legoESM files an unmatched
sampling protocol.
