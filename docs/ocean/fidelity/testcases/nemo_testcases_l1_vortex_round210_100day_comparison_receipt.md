# Round 210 — VORTEX vs NEMO over NEMO's shipped run length (100 days)

**Verdict: HELD, measurement only.** Both certified 30 km cards (flux-form UP3,
vector-invariant) were run from rest for NEMO's own shipped run length
(`nn_itend = 3000` steps of `rn_Dt = 2880` s = 100 days) on both legoESM and
NEMO, scored daily. No physics or option change; nothing lands except the
scoring script, this receipt, and the evidence under `phase3/round210/`.

**Headline.** Both cards' kt=10 debt stays harmless at a year's shipped run
length, but the two cards' error GROWS WITH A DIFFERENT SHAPE. The flux card's
known kt=2 debt (`u` 1.2168e-08 normalized) compounds into a **~10⁴x** growth
over 100 days (T rms 4.90e-10 K → 9.04e-06 K, roughly `day^2.1`–`day^2.4`) —
superlinear, consistent with a systematic bit-level difference being advected
by the flow, not pure rounding noise. The vector card, already at the bar at
kt=2, grows only **~300x** over the same 100 days (T rms 6.04e-15 K →
1.83e-12 K, roughly `day^1.2`–`day^1.4`) and stays within ~2000x of the
double-precision floor throughout — consistent with accumulated rounding, not
a systematic difference. Both are many orders of magnitude below anything a
10-day or 30-day ladder, or a physical judgement of the flow, would call
significant.

---

## 1. What ran, and what in it is NEMO's own pin vs this round's choice

Both NEMO runs used the two already-certified executables (no rebuild; same
grid-size-derived-at-runtime arrangement round 208 established), in a **fresh
run directory** copied from each build's `EXP00` (the certified `EXP00`
directories, which pin `nn_itend=10`/`nn_stock=10` for the kt-ladder, were not
touched). The resolved namelist for each round-210 run directory was diffed
against `tests/VORTEX/EXPREF/namelist_cfg` (the shipped deck NEMO ships with
the test). Flux:

```
cn_exp      "VORTEX_OMIP_L1_ZCO"  vs EXPREF "VORTEX"        (label only)
nn_itend    3000  vs EXPREF 3000                             (IDENTICAL value)
nn_stock    30    vs EXPREF 99999 (no restarts)
```

Vector adds exactly the one already-certified decision-73 delta
(`ln_dynadv_vec=.true.`/`ln_dynadv_up3=.false.`, the ORCA2/GYRE vector-invariant
momentum scheme) — nothing new. **`nn_stock = 30` is this round's own choice,
not EXPREF's**: EXPREF writes no restart at all. It selects the OUTPUT cadence
(a restart dump every 30 steps = daily, since `rn_Dt=2880` s gives 30
steps/day) and does not touch any term the model integrates — restart writing
is a pure read of the current prognostic state, so it cannot change the
trajectory. `ln_meshmask` was also set to EXPREF's own `.false.` (dropped from
the ladder's diagnostic `.true.`; the mesh file is not used by this round's
scoring, which reads masks from the card's own Python-side construction, the
same convention the certified kt-ladder already uses). No other line differs.
Both runs used `mpirun -np 1 --oversubscribe ./nemo` (single rank, matching the
ladder's acquisition pattern), one at a time.

legoESM ran the certified card unmodified (`build_nemo_testcase_card`,
production `model_config`, no test hooks) for 3000 `model.step` calls in one
continuous in-process loop — no restart chaining was needed or used, so the
"prove the chain is bit-identical" clause does not apply.

## 2. Reused, not re-derived (pre-impl search)

Grepped `scripts/validate/ocean_fidelity/testcases` for `year`, `daily`,
`snapshot`. Two existing pieces were extended rather than duplicated, into
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round210_100day_comparison.py`:

- `nemo_testcase_phase3_trajectory_gate.lego_fields` / `expected_masks` — the
  SAME field extraction and wet-face masks (T/S/u/v active; `ssh` wet) the
  certified kt=1..10 ladder already uses for these two cards.
- `nemo_testcase_l2_gyre_year_fromrest.py`'s NEMO-restart reader pattern
  (`_load_nemo`): same variable names (`tn`/`sn`/`un`/`vn`/`sshn`), same axis
  contract `(time_counter, nav_lev, y, x)` → `transpose(1, 2, 0)`, same
  finite-value check. VORTEX's restart carries **no halo** (interior is
  exactly 63×63, matching the card's own shape) — read from the header, not
  assumed.

VORTEX's own wet mask excludes a one-cell closed-boundary frame
(61×61 = 3721 wet columns of 63×63 = 3969): confirmed by printing
`expected_masks` cell counts before scoring, not assumed.

## 3. Sanity check: this round's first 10 steps reproduce the certified ladder

Before the 100-day run, the trajectory gate's own `run()` was pointed at this
round's fresh NEMO output directories (whose unmodified `stprk3.F90` writer
still fires the existing `oracle_step_entry_kt*.bin` instrument for kt=1..60 —
not edited for this round) at `max_step=10`, and its rows were required to
equal the certified round-207/208 registries row for row.

```
flux: kt1-10 sanity vs certified ladder: REPRODUCED
vec:  kt1-10 sanity vs certified ladder: REPRODUCED
```

Both REPRODUCED exactly (this is the mechanical gate, not a visual check — the
script raises if any row differs). This makes the day-0.33 point **identical
to the existing certified kt=10 row** (kt=10 = 9 completed `model.step` calls,
exactly `10/30` of a day), satisfying the round's sanity requirement without
re-deriving it: it is the already-certified number, reused.

## 4. Spot checks (by hand, outside the script)

Two table entries were recomputed independently from the raw snapshot/restart
files, applying the same mask, and matched the script's reported value
exactly:

- **flux, day 1, T**: hand `rms = 4.896477150795794e-10`, `max = 1.9775093562657275e-08`
  — script: `4.896477e-10` / `1.9775e-08`. MATCH.
- **vector, day 30, u**: hand `rms = 3.9126023346672885e-14`, `max = 3.898478762032198e-13`
  — script: `3.9126e-14` / `3.8985e-13`. MATCH.

(An earlier unmasked hand check at day 1 gave `4.741e-10`, which looked close
but was wrong — it included the closed-boundary frame; re-running it with
`expected_masks` resolved the discrepancy and is the number quoted above.)

## 5. Table (T/u/ssh rms; full T/u/v/ssh rms+max in `round210_scores.json`)

| day | flux T rms [K] | flux u rms [m/s] | flux ssh rms [m] | vec T rms [K] | vec u rms [m/s] | vec ssh rms [m] |
|---|---|---|---|---|---|---|
| 1 | 4.896477e-10 | 4.762378e-10 | 2.143053e-10 | 6.036321e-15 | 2.380686e-15 | 1.225320e-15 |
| 2 | 4.908651e-10 | 6.143341e-10 | 2.459680e-10 | 7.877150e-15 | 3.523345e-15 | 1.555369e-15 |
| 5 | 6.123451e-08 | 1.018113e-07 | 4.099634e-08 | 1.409647e-14 | 6.617719e-15 | 2.461152e-15 |
| 10 | 1.236815e-07 | 2.703513e-07 | 9.561490e-08 | 2.379238e-14 | 1.172479e-14 | 3.758521e-15 |
| 20 | 5.104381e-07 | 1.481468e-06 | 7.883451e-07 | 4.485488e-14 | 2.237288e-14 | 6.723581e-15 |
| 30 | 9.661615e-07 | 2.336168e-06 | 2.235551e-06 | 7.645451e-14 | 3.912602e-14 | 1.402938e-14 |
| 60 | 3.674024e-06 | 1.353608e-05 | 7.726718e-06 | 3.401017e-13 | 1.843014e-13 | 1.037212e-13 |
| 100 | 9.037083e-06 | 3.205187e-05 | 2.825655e-05 | 1.830225e-12 | 1.361880e-12 | 7.429940e-13 |

Day-100 maxima (full-precision in the JSON): flux T max 1.0295e-04 K, u max
5.4341e-04 m/s, v max 3.1275e-04 m/s, ssh max 1.1159e-04 m; vector T max
3.7751e-11 K, u max 5.0293e-11 m/s, v max 5.7481e-11 m/s, ssh max 1.4895e-11 m.

Curves (log scale, both cards, T/u/ssh rms, days 1-100):
`phase3/round210/round210_curves.png`.

## 6. What the ten-step floor becomes at 100 days, and the shape comparison

The flux card's first-over-bar debt (kt=2 `u`, normalized 1.2168e-08 — note
CA/CB) is not a one-step artifact: by day 100 its raw T/u/v/ssh rms have grown
**~1.8e4x** from day 1, fit over the whole run as `day^2.1`–`day^2.4`
(T and u respectively) — superlinear, the signature of a systematic
difference being carried and amplified by the flow rather than random
rounding. The vector card's day-1 values already sit at the double-precision
floor (~1e-15 K) and grow only **~300x** by day 100, fit as `day^1.2`–`day^1.4`
— close to the accumulation rate of pure rounding noise, not a systematic
term. **The two cards' curves differ in shape, not just in magnitude**: this
is visible in the log-scale PNG as a steeper flux slope throughout.

In absolute terms neither is large: day-100 flux T rms is 9.0e-6 K and ssh rms
is 2.8e-5 m, both far below anything resolvable in a physical comparison at
this resolution.

## 7. GYRE year context (different case — not a ratio)

GYRE's own from-rest year run (certified, round 207, note BZ) has day-30 T3D
rms = 2.3432465132112266e-06 K. VORTEX's flux card's day-30 T rms
(9.661615e-07 K) is roughly the same order of magnitude; the vector card's
(7.645451e-14 K) is eight orders smaller. **This is reported as context only**:
GYRE is a doubly-forced wind-driven double-gyre with TKE vertical mixing and a
different domain/physics set entirely; VORTEX is an unforced idealized vortex
spin-down with constant mixing. No ratio between the two cases is a
fidelity statement.

## 8. Unasked choices

- `nn_stock = 30` (daily restart cadence): stated above as this round's own
  pick, not EXPREF's; does not affect the trajectory. Not asked because it is
  pure instrumentation (the round's own brief names this exact example).
- `ln_meshmask = .false.`: set to match EXPREF exactly (dropped from the
  ladder's diagnostic `.true.`), since this round's scoring never reads the
  NEMO mesh file. Diagnostic-output-only; does not affect the trajectory.
- No scientific/physics/scheme choice was made. Both cards' certified
  model_config ran unmodified.

## 9. Evidence, wall time, and command

- NEMO flux: `phase3/round210/nemo_run_flux/` (100 restarts, `STOP 0`,
  **81 s** wall, single rank).
- NEMO vector: `phase3/round210/nemo_run_vec/` (100 restarts, `STOP 0`,
  **71 s** wall, single rank).
- legoESM flux: `phase3/round210/lego_flux/day{001..100}.npz`, CPU, fp64,
  ~3 min wall (in-process, no JIT recompilation across the 3000 steps).
- legoESM vector: `phase3/round210/lego_vec/day{001..100}.npz`, CPU, fp64,
  ~5 min wall.
- Full daily series + sanity/cross-check report:
  `phase3/round210/round210_scores.json`.
- Curves: `phase3/round210/round210_curves.png`.
- Script (committed):
  `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round210_100day_comparison.py`.
- Command: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src .venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round210_100day_comparison.py`

## 10. No reviewer needed

Per the round's order: no code change beyond the scoring script, which runs no
model and changes no option; both models' certified configs ran unmodified.
Reviewed by spot-check instead (§4), per the order's own substitution.

## 11. Open

VORTEX walk remains STOPPED at its milestones (note CA/CB). This round adds no
new candidate and recommends none: both cards' debt is harmless at 100 days.
If a longer run is ever wanted, note the flux card's superlinear shape means
its debt would keep growing faster than the vector card's — worth re-checking
before any multi-year VORTEX use, should one ever be proposed.
