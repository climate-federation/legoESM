# Split-explicit momentum chain — round 94 climate close

Date: 2026-08-30. Session: `01a04e34-d1fb-73e0-b25a-177641f0a246`.

The preregistered current-default climate re-battery passes all three science
bars. This closes the ordered lane. No new GPU or MPI process was launched in
this binding round; it imports the completed host-run artifacts below.

## Receipt admission

The authoritative score is
`/tmp/dino-climate-rebattery-01a04e34/dino_climate_rebattery_score.json`,
SHA-256
`db966e97b214151789e203cc6058d0c0a3564071cc92e301b5ac63aa8b7f693e`.
It stamps schema `dino-current-default-climate-rebattery-v1`, producer
`9548be86181a1f362355e0cdc30ce4ac98d14e3a`, this session, the bridged NEMO
before state, faithful T-point prior stress, bridged TKE, fp64 control and
scored storage, and NEMO's `both` vertical ladder.

| arm | purpose | SHA-256 | duplicate admission |
|---|---|---|---|
| `climate_a` | 360-day MLD/basin member A | `e598d01e3baf69c1af48df3b3b8df66150052820bc13a782a5b8be413c16cb20` | bit-identical to B |
| `climate_b` | 360-day MLD/basin member B | `e598d01e3baf69c1af48df3b3b8df66150052820bc13a782a5b8be413c16cb20` | bit-identical to A |
| `wall_a` | five-day wall member A | `4a4a4bcd0a1a6a77430e437ee76b7301e5a1c95a18c73ca8378d23caf736731c` | bit-identical to B |
| `wall_b` | five-day wall member B | `4a4a4bcd0a1a6a77430e437ee76b7301e5a1c95a18c73ca8378d23caf736731c` | bit-identical to A |

All four arms share initial-state SHA-256
`01ec6db577943529f72a6fb225b8bca4b0af6e1e818f8dfc50b222df619344a8`.
Their complete resolved configs are identical within each registered pair and
select every faithful ZDF, barotropic, momentum and Redi default. The scorer's
MLD, basin and wall classifier plants all fire. The wall-detail receipt,
SHA-256
`50981444479aee4e797aba8e3a1fc89cbc7313fe551bbb58238a9575abb1ccd0`,
also passes land poisoning and both amplitude plants; the measurement-scale
plant recovers `0.9997275` and the large plant `0.9999803` of their injected
signals.

## Registered climate verdicts

| question | current faithful result | frozen bar | verdict |
|---|---:|---:|---|
| southern-basin day-90 MLD RMS | `0.0001037972402 m` (`0.103797 mm`) | confirm at `<=11.2397455 m` | **CONFIRM** |
| day-360 southern-basin transport gap | `-0.02318376361 Sv` | response `>=10%`, outside `2F` | **CONFIRMED** |
| wall first-eight-step ratio/share | `1.199732107 / 0.109118838` | confirm at `<=1.25 / <=0.17` | **CONFIRMED** |

The basin comparator's NEMO frame is `10.01607116576 Sv`. Relative to the
registered historical gap `-0.95191223318 Sv`, the current gap moves by
`+0.92872846957 Sv`: `15.0434` times the frozen
`F=0.06173656216 Sv` floor and `97.5645%` of the old deficit. All five
absolute day-90 acceptance rows pass their 5x gates. The honest result is that
the **combined current faithful default** closes the registered basin deficit;
this four-arm battery does not allocate that closure among the individual
tracer-velocity, momentum-couple, Redi or A33 fixes.

The MLD result supersedes the older faithful-ZDF-only `22.4795 m` refutation
for the current complete default. The wall result likewise supersedes the
older `1.65/0.170` epoch for this complete default. Those older experiments
remain valid historical/intervention receipts, but they are not the present
production epoch.

## Registry close and qualifications

The final full-step result remains **34/37 = 91.891892%** strict active
coverage and `116/116` accounting coverage, with zero uncovered calls. The
three visible active debts remain:

1. `stpmlf.F90:204` `ldf_dyn`, coefficient-isolated measurement;
2. `stpmlf.F90:387` `tra_sbc`, forcing-to-tracer-RHS application;
3. `stpmlf.F90:394` `tra_qsr`, two-band vertical redistribution.

Round 93's `78.S.3` salinity `zfw` remains explicitly
`CLEARED-RULE-1B`: 212/9,920 columns, maximum normalized error
`6.690652e-15`, below the frozen `2e-14` proven-oracle-arithmetic ceiling.
Neither the climate confirmation nor registry harvest relabels it bit-exact.

The climate claims are scoped to the state-initialized DINO twin: it starts
from NEMO's developed day-180 restart and bridges before-level state, carried
T-point stress and TKE. They are not yet a claim that a standalone legoESM
DINO recipe starting from its own initial condition reaches the same climate,
and they do not transfer automatically to another recipe. All four battery
arms use the `nemo_dino_kamm_mlf` faithful-default card. The same selectors
default on both DINO NEMO cards, but this battery did not climate-test
`nemo_dino_kamm`; neither standalone initialization nor cross-recipe transfer
is claimed. The registered next campaign step is therefore a standalone-
recipe transfer test, with the three remaining call measurements kept visible
rather than folded into the climate headline.

## Campaign-ledger paragraph for #1455

The completed DINO/NEMO ordered fidelity walk now closes all three registered
current-default climate tests: in the state-initialized, before/T-stress/TKE-
bridged twin, the day-360 southern-basin transport deficit contracts from the
historical `-0.9519 Sv` to `-0.02318 Sv` (`+0.92873 Sv`, `15.04F`, **97.56%**
closure; NEMO frame `10.0161 Sv`), southern-basin day-90 MLD RMS falls from
the earlier ZDF-only `22.5 m` epoch to `1.038e-4 m` (`0.104 mm`), and the
five-day wall-flicker ratio/share is `1.1997/0.1091`; all three preregistered
bars pass with bit-identical duplicate arms: the MLD verdict is **CONFIRM**,
and basin and wall are **CONFIRMED**. All registered
classifier/reducer/wall plants fire, and all provenance/identity gates pass.
The source-ordered
registry is **34/37 = 91.891892%** measured active
coverage (`116/116` accounted, zero uncovered): `ldf_dyn`, `tra_sbc` and
`tra_qsr` remain openly unmeasured, and salinity Redi `zfw` retains its
Rule-1b `6.69e-15` arithmetic-floor qualification rather than a bit-exact
label. These are twin-bridge results for `nemo_dino_kamm_mlf`. The selectors
also default on `nemo_dino_kamm`, but that card, standalone initialization and
cross-recipe transfer were not climate-tested; transfer is the next registered
test.

## Independent dual review

Two independent reviewers audited the closing claims; the author did not
self-review.

- The artifact reviewer independently reproduced every metric, ratio, verdict,
  hash, duplicate identity and bar from the score and wall-detail artifacts.
  Its initial hold found that the draft overextended the MLF-only battery to
  both DINO cards, called all storage fp64 rather than the scored storage, and
  conflated passing provenance gates with firing plants. All three were
  corrected. Final verdict: **PASS / SHIP**, no remaining findings.
- The scope reviewer independently checked the PR and ledger against the
  preregistration, round-93 result and coverage harvest. Its initial hold found
  the same MLF-card boundary, grouped S.3 incorrectly with the three residual
  measurements, and pluralized all verdicts as `CONFIRMED` when MLD's exact
  label is `CONFIRM`. All three were corrected. Final verdict: **PASS**, no
  remaining findings.

Both final reviews affirm the combined-bundle/non-owner interpretation,
historical-epoch treatment, twin-bridge boundary, standalone-transfer next
step, exact three-item measurement debt, and separate S.3 Rule-1b
qualification.
