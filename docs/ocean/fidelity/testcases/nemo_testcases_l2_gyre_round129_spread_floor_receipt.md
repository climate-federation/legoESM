# NEMO testcase L2 GYRE phase 3 — round 129 spread-floor receipt

Date: 2026-09-20

Incoming tip: `2d0665437e00b9e560740a6df92e4b90ee136358`

Status: **HELD — the day-240 GYRE year gap is SYSTEMATIC by the user's
registered rule.  legoESM's maximum pairwise T3D spread is
`2.0891293703252062e-10 K`, only `1.2702390413735022e-08` of the matched
seed-0 gap `1.6446741930292448e-2 K`, far below the `0.3` bar.  NEMO's
spread is `3.304067814752494e-10 K`, `1.5815525173714653` times legoESM's
and therefore inside the registered `[0.5, 2.0]` similarity band.  The
spread is flat through day 240 (`a=0.07823`, day-240/day-30 `0.90958`),
while the gap grows as `t^2.28425`.  No physics or configuration lands.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round129/`

## Outcome first

The user's year-spread bar is **NOT MET**.  Its exact primary ratio is

`R240 = 2.0891293703252062e-10 / 1.6446741930292448e-2`
`= 1.2702390413735022e-08`.

The required `0.3 * gap` spread is `4.934022579087734e-3 K`; the measured
spread is smaller by `23,617,601.902` times.  Equivalently, the gap is
`78,725,339.675` times the legoESM spread.  This refutes the preregistered
prediction that the EVD discontinuity would amplify this exact `1e-10 K`
initial perturbation enough to put the year gap inside the model's spread.

The secondary discriminator also fires.  NEMO/legoESM day-240 spread is
`1.5815525173714653`, within the preregistered `[0.5,2.0]` band.  Therefore
legoESM spread is much smaller than the cross-model gap while NEMO spread is
the same order as legoESM spread: the gap is **SYSTEMATIC** under the user's
rule, not within either model's infinitesimal-IC spread.

The next round must stop cycling through temperature/EVD/vertical-diffusion
attributions and return to a statement: the first non-bit operand in the
dynamics feeding the western jet, using the existing kt records.  It must not
resume last-bit/XLA rounding walks.

## The registered eight-day table

Every spread is the maximum unweighted fp64 T3D RMS over all six pairs among
seeds 0--3, on NEMO's own 18,000-cell wet `tmask`.  Each gap is the matched
legoESM-versus-NEMO seed pair.  No ensemble mean, unmatched pair, depth
weight, region or post-hoc day enters the verdict.

| day | legoESM max pair spread (K) | NEMO max pair spread (K) | gap seed 0 (K) | gap seed 1 (K) | gap seed 2 (K) | gap seed 3 (K) |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | `2.2967947032885092e-10` | `1.443889737136528e-10` | `6.890484901489568e-5` | `6.890486551856493e-5` | `6.890485807283965e-5` | `6.890486132449858e-5` |
| 60 | `1.53383898563116e-10` | `1.5912356044238087e-10` | `1.9329973681936875e-4` | `1.9329973429448532e-4` | `1.93299731401527e-4` | `1.9329973586944805e-4` |
| 90 | `3.0500398233435986e-10` | `2.1442271381253624e-10` | `1.8645021144913585e-3` | `1.864502123944721e-3` | `1.864502111095511e-3` | `1.864502097040921e-3` |
| 120 | `4.1613307710637706e-10` | `1.2831729596957783e-9` | `1.0501256819510476e-3` | `1.0501255892096434e-3` | `1.0501253210408634e-3` | `1.0501255404533475e-3` |
| 180 | `2.338424350360282e-10` | `2.986307953832913e-10` | `3.580551011866709e-3` | `3.5805508980435003e-3` | `3.5805509513096843e-3` | `3.5805509524246097e-3` |
| 240 | `2.0891293703252062e-10` | `3.304067814752494e-10` | `1.6446741930292448e-2` | `1.644674191412724e-2` | `1.6446741923675838e-2` | `1.6446741903163718e-2` |
| 300 | `1.8003533733351255e-10` | `2.5956598277082876e-10` | `1.3597404177319843e-2` | `1.3597404160286462e-2` | `1.3597404174983167e-2` | `1.359740415680497e-2` |
| 360 | `1.5738363873642203e-9` | `3.394534692102701e-10` | `1.1223573910167267e-2` | `1.1223574025829068e-2` | `1.1223573943034056e-2` | `1.122357395738855e-2` |

The pinned NEMO spread is reproduced bit-for-bit on all eight rows.  The
seed-0 gaps at days 30, 240 and 360 reproduce the immutable controls exactly
as binary64 values.  The four matched gaps are nearly identical; at day 240
their full range is `2.7128729818137742e-11 K`, so choosing the registered
seed-0 gap rather than a post-hoc member cannot explain the verdict.

## Growth-shape verdict

The preregistered ordinary least-squares log-log fit to positive legoESM
spread values at days 30, 60, 90, 120, 180 and 240 gives exponent
`0.07822999685979398` with weak `R^2=0.03021342643371494`.  The day-240/day-30
factor is `0.9095847214093747`.  By the frozen labels, `|a| <= 0.25` is
**flat**, so the registered non-flat prediction and the `t^2.3-like`
prediction are both **REFUTED**.

The same scorer independently reproduces the control-gap eight-day exponent
`2.2842519410812514` with `R^2=0.9300224444473821`.  The spread and the gap do
not share a growth mechanism at the registered perturbation amplitude: one is
flat at `1e-10`-class RMS through day 240, while the other grows into the
`1e-2 K` class.

This does not contradict the previous finding that the EVD trigger can amplify
an upstream difference.  It refutes the narrower prediction that these four
specified `1e-10 K` IC members measure a spread large enough to account for
the current seed-0 cross-model trajectory.

## Common-perturbation admission

The four NEMO records all name binary SHA-256
`578c88f17ecaa8052276ff43e6b6c928f5be49fb218d4af33bc8718472613c4a`.
Their namelists select `nn_pert_seed=0,1,2,3`; each runtime log prints the
selected value, seeds 1--3 print the executed `TINY PERTURBATION SEED` marker,
and seed 0 does not.

The compiled selector is declared at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:42`, included in the
read namelist at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:118-122`, and printed
at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:137-145`.
The compiled branch guards on nonzero seed, adds
`1e-10 * sin(NINT(pdept)*73 + NINT(gphit*1000)*179 + seed*997) * ptmask`
to temperature, and prints the runtime marker at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_istate.f90:101-105`.

The step-entry writer records fp64 `ts`, `uu`, `vv` and `ssh` from `Nbb` at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:89-99`.  The committed
gate reconstructs legoESM's four initial states using the existing certified
perturbation transcription and compares against those records:

| seed | T unequal / cells | S unequal / cells | u unequal / cells | v unequal / cells | SSH unequal / cells | max abs over every field |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0 / 18000` | `0 / 18000` | `0 / 17400` | `0 / 17100` | `0 / 600` | `0.0` |
| 1 | `0 / 18000` | `0 / 18000` | `0 / 17400` | `0 / 17100` | `0 / 600` | `0.0` |
| 2 | `0 / 18000` | `0 / 18000` | `0 / 17400` | `0 / 17100` | `0 / 600` | `0.0` |
| 3 | `0 / 18000` | `0 / 18000` | `0 / 17400` | `0 / 17100` | `0 / 600` | `0.0` |

Thus both models receive **bit-identical complete initial states for every
matched seed**, not merely equivalent formulas or equal perturbation norms.
The independently read latitude/depth operands also have zero raw and rounded
disagreements, and the mesh hash is the registered
`3bf5d10e36dc52336b9797b13eb1efb4f02d25e0fb3a0c450ac6ce69e65471df`.

The preregistration's first committed citation named lines 82--90 of
`usrdef_istate.f90`.
That range is the base profile, not the perturbation.  It is explicitly
retracted and corrected to lines 101--105 in the preregistration and here; no
prediction, value, population or falsifier changed.

## Member and time-level provenance

The certified year integrator is untouched.  It is byte-identical at the
operator-directed reused member-0 commit and the new member producer commit,
SHA-256
`7a679711c8ce02191e839f9f4359f21e753e8e2b6014a19a2302fb69ca168f9c`.
The certified phase-3 stepping gate is likewise unchanged, SHA-256
`e57fe1c475a1d386f30856f1841a2efb65162df6968b74b1a200f8850bdd9112`.

| legoESM seed | root | clean producer commit | days / steps / daily snapshots | wall seconds |
|---:|---|---|---:|---:|
| 0 | `phase3/year_equivalence/gyre/lego_seed0_year` | `4d250301588d3ed0ad83fb20d6bf520e175d576e` | `360 / 2160 / 360` | `1957.859` |
| 1 | `phase3/round129/lego_seed1_year` | `007affce297763a9594aea19babf8c9eace4883a` | `360 / 2160 / 360` | `2196.061` |
| 2 | `phase3/round129/lego_seed2_year` | `007affce297763a9594aea19babf8c9eace4883a` | `360 / 2160 / 360` | `2184.066` |
| 3 | `phase3/round129/lego_seed3_year` | `007affce297763a9594aea19babf8c9eace4883a` | `360 / 2160 / 360` | `2188.357` |

The operator's `year_owners` shorthand contains the seed-0 trajectory used by
other owner gates.  The four-member NEMO ensemble used for the registered
NEMO-vs-NEMO floor is, as its original receipt records, under
`phase3/year_fromrest/nemo_seed0` through `nemo_seed3`; those are the paths
scored here.

NEMO completes stage 3, swaps `Naa` into `Nbb`, and writes the restart from
that completed level at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:215-222` and
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:260`.
The restart writer maps `sshn`, `un`, `vn`, `tn` and `sn` from `Kbb` at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/restart.f90:176-180`.
Therefore a restart stamped step `6*day` and a legoESM state after the same
number of `model.step` calls are the same completed-step time level.

The authoritative score is `round129/spread_floor.json`, SHA-256
`f8c9e9812e5cba957e8b96f7e3f840f6bba624c3d32e2bff49edb2962f7684ba`.
It was emitted at clean scorer commit
`d123cbafd4a839e812a9cbb3fcd54dd748270067` under CPU, fp64 state/geometry,
and the fp64/libm precision policy.  It carries every scored input path and
SHA-256, all 96 pair distances (six pairs x two models x eight days), the 32
matched gaps, source and manifest hashes, initial-state rows and worktree
stamp.  `round129/spread_floor.log` is the human-readable table.

## Frozen prediction ledger

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round129.md`, committed
before any new member was run as
`007affce297763a9594aea19babf8c9eace4883a`.

- **P0 CONFIRMED.** All five initial fields are bit-exact for every matched
  seed; the source branch, runtime selectors and one-binary ensemble also
  admit.
- **P1 CONFIRMED.** The reused seed 0 and new seeds 1--3 have clean producer
  stamps, the exact registered commits, byte-identical year/stepping gates,
  2,160 steps and all 360 daily snapshots.
- **P2 CONFIRMED.** The population is 18,000 wet T cells; all six pairs exist
  for each model on every registered day; all four matched gaps exist.  The
  pinned NEMO spread and seed-0 gap controls reproduce exactly.
- **P3 REFUTED.** `R240=1.2702390413735022e-08`, not `>=0.3`.  Because the
  NEMO/legoESM spread ratio is `1.5815525173714653`, the user's systematic-gap
  discriminator is satisfied.
- **P4 REFUTED.** The spread day-240/day-30 factor is `0.9095847214093747`,
  not `>10`, and its exponent is `0.07822999685979398`, classified flat rather
  than `t^2.3-like`.
- **P5 CONFIRMED.** No production physics, card, configuration, carried state,
  restart schema, stabilizer or certified year harness changes.

## Plant controls

The scorer self-check proves the exact binary64 `0.3` boundary: the adjacent
value below is NOT_MET, equality is MET, and the adjacent value above is MET.
It also recovers a synthetic `2.3` exponent and proves that the six-pair
registry can fail.  Its log is `round129/scorer_self_check.log`.

The initial-state plant changes one real wet seed-1 temperature value by one
binary64 step.  It reports `seed 1 ... not BIT on ['T']`, prints
`STATUS PLANT-FIRED`, and exits 1 in
`round129/plant_initial_temperature_ulp.log`.

The pair-registry plant removes the real `(2,3)` pair.  It prints the five
observed pairs and six required pairs, prints `STATUS PLANT-FIRED`, and exits 1
in `round129/plant_pair_registry.log`.  Neither plant perturbs a zero, dry cell
or unconsumed row.

## Spread is not literal run-to-run nondeterminism

This receipt follows the user's specified `nemo_istate_perturbation` ensemble
bar.  It does not relabel it as stochastic run-to-run variability.  The prior
same-binary seed-0 reproducibility arm was bit-identical on every saved field
at all scored days, including exactly `0.0 K` T3D RMS.  The literal
same-binary run-to-run floor is therefore zero; the `1e-10 K` initial-condition
spread is the controlled sensitivity floor available here.  Even that larger
floor misses the day-240 gap by 78.7 million times.

## Landing and cross-card scope

No physical statement lands, so the immutable GYRE before arm remains
`phase3/year_equivalence/gyre/` and all certified ladder/month/year headline
rows remain unchanged.  The only executable addition is a read-only scorer
that imports, but does not alter, the certified year harness.

GYRE, generic NEMO-GYRE, DINO, LOCK_EXCHANGE and OVERFLOW execute no changed
production statement.  ORCA2 is **UNMEASURED-WITH-SPEC**: run the same four
matched seed perturbations for 360 days, score its native wet T3D population
at the eight registered days, and apply the same exact pair registry and
initial-state admission before transferring this verdict.

No configuration or carried-state question is exposed.  `DECISION_NEEDED` is
`NONE`.

## Independent adversarial review

The required separate pass was invoked with `codex exec --sandbox read-only`
against the complete incoming-tip diff, authoritative score, member manifests,
both plants, pinned NEMO verdict and compiled YRPERT sources.  It was asked to
refute the initial-state identity, split-root provenance, six-pair table,
exact verdict boundary, systematic label, growth label and next-round
instruction.  Its verbatim terminal result was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

codex_exit_code=1
```

Independent review was unavailable in-sandbox.  It emitted neither `SHIP` nor
`DO NOT SHIP`; no verdict is fabricated.  The complete log is
`round129/codex_review.log`.

## Verification

The scorer self-check, Python byte compilation and `git diff --check` pass.
The two record-backed plants each print `STATUS PLANT-FIRED` and exit 1, as
described above.

The clean-tree focused suite covers the new synthetic boundary/pair/growth
controls, the full record-backed score, both subprocess plants, every citation
map audit and the worktree-stamp ratchet.  Its exact summary is:

```text
============================= 34 passed in 15.66s ==============================
```

The log and JUnit report are `round129/focused_tests.log` and
`round129/focused_tests.xml`.

The standard four-file integration/push gate covers the complete citation
suite, NEMO TKE terms, resolved NEMO recipes including the forced GYRE
trajectory, and freshwater closure.  Its exact summary is:

```text
======================= 119 passed in 369.87s (0:06:09) ========================
```

The log and JUnit report are `round129/four_file_push_gate.log` and
`round129/four_file_push_gate.xml`.

The receipt citation gate finds 8 citations, zero unmapped citations, zero
failures and zero map-audit failures in `round129/citation_gate.json`.
Shifting the compiled perturbation citation by two lines produces
`SYMBOL-NOT-AT-LINE` on its first endpoint, prints no PASS status and exits 1;
its report and log are `round129/citation_gate_shifted_plant.json` and
`round129/citation_gate_shifted_plant.log`.

## OPEN — round 130

The year gap is **SYSTEMATIC** under the exact user rule.  Round 130 returns to
producing a statement, not another attribution cycle:

1. Use the existing kt records and current production closure to isolate the
   dynamics feeding the western jet where the macroscopic temperature gap is
   born.  Start from the first recorded boundary that feeds that jet and name
   the first non-bit **operand statement**, in compiled execution order.
2. Require production JIT and eager rows plus a production plant.  Do not
   pursue one-ULP/XLA association residues whose magnitude is orders below the
   day-240 gap.
3. Preserve the exact seed-0 day-30/day-240/day-360 controls and the flat
   ensemble-spread result as discriminators.  A proposed owner must carry a
   macroscopic share of the western-jet/year gap, not merely alter an EVD
   trigger after the fact.
4. No NEMO acquisition is expected: use the existing kt/stage/operator records
   first.  Request a new record only if the first required consumed operand is
   absent, and name that operand and compiled write site in a fail-closed
   `run.sh`.
5. Any eventual production candidate still faces the Decision-43 ladder and
   month criteria plus the Decision-45 360-day year gate, against a same-tip
   before arm, with DINO measured when the statement is shared.

The recommended campaign target remains GYRE's western-jet dynamics until a
source statement owns this systematic year gap.  Do not move to DINO or ORCA2
on the false premise that GYRE met its spread bar.
