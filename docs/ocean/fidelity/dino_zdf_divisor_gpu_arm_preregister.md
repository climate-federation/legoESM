# Preregistration — the DINO zdf-divisor GPU arm

Branch `fidelity/dino-zdf-divisor-scaling`, arm launched at tip `0d1d49c26`
(the fix is `d80df3ed7`, "NEMO's `e3w(Kmm)` is the unbranched implicit-mixing
divisor").  Written **before** either GPU run finished.  Companion receipts:
`dino_zdf_divisor_scaling.md` (the measurement), `dino_zdf_divisor_arm_receipt.md`
(the CPU arm), `dino_multi_year_climate_equivalence_result.md` (the capstone this
arm is scored against).

The question, in one line: **does the implicit-mixing divisor own the 20-year
climate divergence the capstone measured?**

Two runs, launched together, one GPU each:

| run | GPU | protocol | baseline it is compared to |
|---|---|---|---|
| year-1 climate battery | 0 | blocks 2/3/4 of `dino_climate_rebattery_round94_handoff.md` | round-94 receipt, producer `9548be86181a` |
| 20-year member m0 (control, unperturbed) | 1 | `/tmp/cap_b5.sh` `run_member(member=0, seed=control)` | capstone `arms/m0.npz`, producer `ddd3a8476afd` |

Output root `/data/abyssal/dbalwada/dino-zdf-divisor-arm/`, `year1/` and
`member20y/`, each with a `run.json` carrying the git SHA, every flag, every
environment variable and the start time, and a `DONE`/`FAILED` marker.

## 1. The one-variable proof — RUN FIRST, and it PASSES

The capstone's six legoESM members were produced at `ddd3a8476afd`.  There are
**188 commits** between that and the fix's parent `9070cf276`, and one of them
edits the twin runner itself (`kamm_twin_90d.py`, +140/-9, adding the
`--bridge-omega` and `--surface-stress-implicit` flags).  So "the capstone's m0
is the before arm" is a CLAIM, and it was tested before anything was launched.

Instrument: the DINO 5-day reach gate
(`docs/ocean/fidelity/testcases/dino_reach_check.md`) —
`kamm_twin_90d.py nemo_dino_kamm_mlf <out>.npz --days 5 --bridge-before`,
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both`, byte-identical
invocation, one variable: the commit.  Four temporary worktrees, four runs,
then every array in the four archives diffed.

| pair | what it answers | key sets | worst numeric `max abs` difference |
|---|---|---|---|
| `ddd3a8476afd` (capstone producer) vs `9070cf276` (fix parent) | is the capstone's m0 a valid "before"? | 8 records added at the parent | **`0.000000e+00`** |
| `9548be86181a` (battery producer) vs `9070cf276` | is the round-94 battery a valid "before"? | same 8 records added | **`0.000000e+00`** |
| `9548be86181a` vs `ddd3a8476afd` | control, the two historical producers | identical | **`0.000000e+00`** |
| `9070cf276` vs `0d1d49c26` | **the fix itself** | identical | `sst 1.259e-02 K`, `u 1.698e-02 m/s`, `v 3.069e-03 m/s`, `eta 3.216e-05 m` |

**VERDICT: the arm is ONE-VARIABLE.**  Both the capstone's `m0..m5` and the
round-94 battery are valid "before" arms; the 188 intervening commits move no
DINO number over 160 leapfrog steps.  The fix's own signature reproduces the CPU
receipt (that receipt's per-day surface `max abs dT` sequence peaks at
`1.26e-02 K` on day 4, which is exactly the all-day maximum recorded here).

The eight keys the parent adds — `bridge_omega_mode`, `bridge_omega_rad_s`,
`config_omega_rad_s`, `bridge_f_T/u/v_sha256`, `een_metric_weighting`,
`surface_stress_implicit` — are RECORDS, not physics: every numeric array is
identical across them, so the two new flags' defaults are measured inert on this
lane rather than assumed inert.

BLIND SPOT, stated because `0.000000e+00` invites over-reading: the twin archive
stores `eta`/`sst`/`u`/`v` as float32 SURFACE slices at five daily samples.  The
proof resolves a difference only to about `1e-7` relative and only at the
surface over 160 steps.  What carries it is that number **together with** the
byte-identical resolved configuration below — not the surface fields alone.  It
is not a claim about ACC, and not a claim about the 3-D interior.

## 2. Resolved-config diff

Same 5-day invocation at the capstone producer and at the fix tip; only rows
that DIFFER are listed.

| field | capstone producer `ddd3a847` | fix tip `0d1d49c2` |
|---|---|---|
| `bridge_omega` | *absent* | `nemo` |
| `daily_acc` | *absent* | `False` |
| `perturb_baro` | *absent* | `None` |
| `perturb_baro_key` | *absent* | `dU_avg` |
| `perturb_baro_scale` | *absent* | `1.0` |
| `perturb_baro_sha256` | *absent* | `None` |
| `surface_stress_implicit` | *absent* | `None` |
| `u_m` | *absent* | `None` |

All **49** fields present on both sides are equal.  Every added row is a field
that the record grew, not a physics selection that changed — three of them
(`perturb_baro*`) were already separate top-level keys in the capstone's own
`m0.npz`, and `daily_acc` is an output flag.

The 20-year member's own config is capstone `m0`'s config plus exactly these
rows: `m0` and the 5-day reach run at the SAME commit differ in only four rows —
`n_days`, `snap_days`, `save_3d`, `bridge_tke` — all invocation rows of two
different runs, with all 43 physics-selector rows equal.

**Named rather than buried: the divisor is in NO config record.**  The fix
REMOVED a field (`implicit_vmix_e3t_now_divisor`) instead of adding one, and
that field was never in this record.  A resolved-config diff is therefore
*blind* to the change under test.  The commit is the only record, which is
exactly why the one-variable proof above had to be an array diff.

## 3. Predictions — 20-year member

Frozen floors and NEMO family means come from
`dino_multi_year_climate_equivalence_score.json` (six legoESM members against
six NEMO members).  For each family the decisive statistic:

| family | frozen verdict | decisive statistic | frozen gap | frozen floor | frozen R |
|---|---|---|---:|---:|---:|
| `acc_series` | UNRESOLVED | `acc.full.month09` | `-4.511e-01` | `4.998e-01` | 0.903 |
| `basin_row_transports` | REFUTE | `row.190.mean` | `-4.712e-03` | `5.479e-05` | 86.0 |
| `density_contrasts` | UNRESOLVED | `density.deep.month04` | `7.113e-04` | `5.306e-04` | 1.34 |
| `mld_seasonal_cycle` | REFUTE | `mld.north of band.month05` | `-3.164e-02` | `1.057e-02` | 2.99 |
| `ts_water_mass_census` | REFUTE | `north of band.abyss_ge1400m.S_mean.mean_y16_y20` | `-1.092e-05` | `1.046e-07` | 104 |
| `variability` | REFUTE | `census.north of band.abyss_ge1400m.S_mean.deseasonalized_std` | `-7.718e-07` | `8.001e-08` | 9.65 |

**If the divisor is the owner** (the scaling study's hypothesis: the removed
error was `+0.30%` in the median and `+12.7%` at the deepest wet level, smallest
in the mixed layer and largest in the abyss, so the implicit vertical coupling
was too weak exactly where the abyssal census gap lives):

1. **`ts_water_mass_census` moves INSIDE the floor.** The abyssal
   `north of band.abyss_ge1400m` `S_mean` and `T_mean` rows collapse toward
   NEMO by order their whole present gaps (`1.09e-5 g/kg`, `2.28e-4 K`).
   Quantitatively, "inside" means the single-member distance
   `|new_gap| <= 2 x floor = 2.09e-07 g/kg` on the decisive row — a **98%**
   contraction.  This is the strongest, most falsifiable row on the board and
   the one this arm is really about.
2. **`mld_seasonal_cycle` moves, but LESS.** The divisor error is 3-4x weaker in
   the mixed layer, so predict a partial contraction of `mld.north of band.
   month05` from `-3.164e-02 m` toward, but not necessarily inside,
   `2 x floor = 2.11e-02 m`.  A move of the right sign that stops short of the
   floor is a PASS for this prediction, not a failure.
3. **NO PREDICTION for `basin_row_transports` or `variability`.** Neither has a
   linear translation from a mean-tendency bias.  Stated up front so that
   whatever they do afterwards cannot be read as confirmation.
4. **`acc_series` and `density_contrasts` must NOT get worse.** They are
   UNRESOLVED at `R = 0.90` and `R = 1.34`; the divisor should not push either
   past `R = 2`.

**REFUTE conditions**, fixed now:

* the abyssal census decisive row does **not** contract by at least `50%`
  (`|new_gap| > 5.46e-06 g/kg`) — the divisor is then not the owner of the
  water-mass gap, whatever else it fixed;
* any currently-UNRESOLVED family's decisive row moves to `R_single > 2` in the
  wrong direction — the fix trades one defect for another;
* the run is unstable, or its `stable=True` / `blew_up_at_step=-1` receipt does
  not fire.

**A move in the predicted direction is still NOT an attribution.**  Attribution
needs the term removed from BOTH models, and the twin no longer has a midpoint
arm reachable from a NEMO card.  The honest ceiling on this arm is "consistent
with the divisor owning it", never "the divisor owned it".

### How a SINGLE member is scored against six-member floors

`member20y/score.sh` runs the capstone's own scorer
(`multi_year_climate_equivalence.py`, sha256 `4f2602ca...`, committed at
`a2ca7e97c`) with its producer pin moved to the new commit — the one guard this
arm deliberately overrides, printed loudly at run time; every other
`validate_lego` guard still binds.

The preregistered family verdicts come from a 20,000-draw bootstrap over six
legoESM members.  **One member cannot produce that verdict**: with a single draw
the legoESM-side spread is undefined and so is the bootstrap.  The script
therefore emits a distance-vs-floor table only —

```
new_gap  = new_member_value - nemo_family_mean       (nemo mean frozen)
R_single = |new_gap| / floor                          (floor frozen, 6+6)
```

— beside the capstone's own ensemble gap and `R` for the same statistic.
`R_single` is **not** the preregistered `R` and carries **no** verdict; it
answers one question, "did this statistic move toward NEMO, and by how many
floors".  A family verdict at this commit needs five more members.  The
per-statistic `moved_toward_nemo` count is reported against its coin-flip null.

## 4. Predictions — year-1 battery

Three registered bars, with the round-94 numbers this arm is compared against:

| bar | round-94 measured | frozen bar | prediction |
|---|---:|---|---|
| southern-basin day-90 MLD RMS | `1.038e-04 m` | CONFIRM at `<= 11.2397455 m` | still CONFIRM, four orders of margin |
| day-360 southern-basin transport gap | `-0.02318376361 Sv` | response `>= 10%`, outside `2F`; `F = 0.06173656216 Sv`, NEMO frame `10.01607116576 Sv` | still CONFIRMED; the gap is already inside one floor, and one year is far too short for the abyssal divisor signal |
| wall first-eight-step ratio / share | `1.199732107 / 0.109118838` | CONFIRM at `<= 1.25 / <= 0.17` | the row most at risk: the fix moves 5-day surface `u` by `1.7e-02 m/s`, and the ratio bar has only `4%` headroom |

A year-1 REFUTE on any of the three is a finding in its own right: the divisor
would then be visible at one year, which the scaling study did not predict.

**Deviation from the round-94 protocol, disclosed:** round 94 ran arm A on GPU 0
and arm B on GPU 1.  Here all four arms run sequentially on GPU 0, because GPU 1
carries the 20-year member.  Round 94 measured A and B **bit-identical** across
those two V100S, so device identity is already known inert; the duplicate-identity
admission gate is unchanged and still has to fire.

## 5. Choices

| choice | status |
|---|---|
| run the two GPU arms, one per GPU | **ASKED** (the user's decision) |
| prove one-variable with the 5-day reach gate at four commits before launching | **ASKED** (task brief) |
| put the three proof worktrees on `/data/abyssal` rather than `/tmp` | **UNASKED**, operational: `/tmp` had 14 GiB free. No scientific content |
| run all four year-1 arms on GPU 0 instead of splitting across two GPUs | **UNASKED**, forced by the GPU allocation; the deviation and the receipt that makes it inert are named in section 4 |
| override the scorer's producer pin for the single new member | **UNASKED**, forced: the frozen scorer refuses any member not stamped `ddd3a8476afd`, which is every member this arm can produce. Printed at run time; no other guard relaxed |
| score the single member as a distance-vs-floor table rather than a family verdict | **UNASKED**, forced by the preregistered bootstrap needing an ensemble; the limitation is stated in the artifact itself, not only here |
| the fourth reach run (`0d1d49c26`) beyond the two the brief named | **UNASKED**, additive: it is the control that shows the reach gate can see the fix at all, which is what makes the two `0.0` rows meaningful rather than vacuous |

---

# RESULT — year-1 battery (added after the run; everything above was frozen first)

Preregistration committed `f2920ee71` at **21:14:39**; the first battery arm was
written at **21:24:32** and the score at **21:41:41**, same day.  Nothing above
this line was edited afterwards.

Score `year1/dino_climate_rebattery_score.json`, sha256
`9a81a6fbb64f12e434da3cd6e421ffe4634c5092c295d29dd700058dedf58ac6`.
`epoch_duplicate_identity = true`: `climate_a` and `climate_b` are bit-identical
(`4b7912fa...`), and so are `wall_a`/`wall_b` (`3dda420b...`), which retires the
one-GPU deviation of section 4 — the duplicate gate fired exactly as it did on
two GPUs in round 94.  All four arms share round-94's initial state
`01ec6db577943529f72a6fb225b8bca4b0af6e1e818f8dfc50b222df619344a8`.

| bar | round-94 `9548be86` | divisor tip `0d1d49c2` | movement | frozen bar | verdict |
|---|---:|---:|---:|---|---|
| southern-basin day-90 MLD RMS | `1.037972e-04 m` | `1.104221e-04 m` | `+6.6e-06 m`, away | CONFIRM at `<= 11.2397455 m` | **CONFIRM**, unchanged |
| day-360 southern-basin transport gap | `-2.318376e-02 Sv` | `-2.521523e-02 Sv` | `-2.03e-03 Sv`, away | outside `2F`, `F = 6.173656e-02 Sv` | **CONFIRMED**, unchanged |
| wall first-8-step ratio | `1.199732107` | `1.199522305` | `-2.1e-04`, safer | `<= 1.25` | **CONFIRMED**, unchanged |
| wall first-8-step share | `0.109118838` | `0.108997294` | `-1.2e-04`, safer | `<= 0.17` | **CONFIRMED**, unchanged |

**All three registered bars pass, none of them moves resolvably.**  The two
"away from NEMO" movements are `0.033` of the transport floor and `6e-07` of the
MLD bar; the NEMO southern-basin MLD frame is `3243.79 m`, so that row moved by
`2e-09` relative.  The wall row, flagged in section 4 as the one at risk because
the fix moves 5-day surface `u` by `1.7e-02 m/s`, moved the *safe* way and kept
`4%` headroom.

**Reading, stated at the ceiling the measurement supports:** the divisor is
INVISIBLE at one year on every registered year-1 metric.  That is what section 4
predicted and it is the honest null — one year is far too short for an abyssal
mixing bias, and a year-1 null neither supports nor damages the 20-year
hypothesis.  Nothing here is evidence about the water-mass census.

New row with no round-94 counterpart, recorded not interpreted: channel MLD RMS
`6.7085 m` against a NEMO channel frame of `256.21 m`; equatorial MLD RMS
`2.34e-07 m`.  The round-94 receipt published only the southern-basin row, so
these two have no "before" and are not compared.

## 20-year member — launch record, still running

Launched on GPU 1 at the same commit, `nohup`, control member (no
`--perturb-seed`), `member20y/run.json` carrying the SHA, every flag, the
resolved config and the config diff.  Measured rate `1.793 s/day` over the first
2350 days; projected total `13011 s` (`3.61 h`), against the capstone m0's
`15487 s` for the same 230,400 leapfrog steps.  Writes
`member20y/m0_divisor.npz` with the capstone key schema and a `DONE`/`FAILED`
marker.  Score it with `member20y/score.sh`.

**Two controls on the scoring instrument were run before the member exists:**

* the frozen scorer's `reduce_legoesm` re-run on the capstone's OWN `m0`
  reproduces the capstone reduced archive's member-0 column **bit-exactly**,
  766/766 statistics, worst absolute difference `0.000000e+00`;
* the same run is also the **null** for the single-member table, and it moves
  the goalposts: a member of the *unchanged* ensemble already scores
  `213/766` statistics "toward NEMO", not `383`, and `R_single` for
  `acc_series` and `density_contrasts` inflates from `0.903`/`1.34` to
  `1.565`/`1.821` from single-member scatter alone.  The abyssal census row does
  NOT inflate (`104.0` against the ensemble's `104.4`), which is why one member
  is informative there and not in those two families.
  Full null in `member20y/NULL_README.txt`.

Two early cross-checks, both clean: the 20-year member and the year-1
`climate_a` arm — independent runs on different GPUs with different output
flags — agree to every printed digit at days 10/30/60; and both differ from
capstone `m0` only in `max|u|`, in the fourth decimal, first visible at day 30.
