# DINO/NEMO fidelity campaign (#1226) — current state digest

**Purpose.** One short file agent briefs can point at instead of re-typing context.
Authoritative detail lives in the gate script's row provenance and in the campaign
memory addenda; this is the map, not the territory. **Update it each iteration.**

Gate (run it, do not quote from here — this line goes stale):
```
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py
```
Last observed 2026-07-30: `AT BAR 15 | DEBT 32 | UNMEASURED 5 | WAIVED 1 | total 53`.
Bar: `corr >= 1-1e-9`, `|ratio-1| <= 1e-6`, per-element `<= 1e-9`.

---

## THE RULE (human, 2026-07-30)
Never guess. Every fix is a **transcription from NEMO source with `file:line` cited**.
If an exact NEMO match does not clear the row or the behaviour, **PAUSE and ESCALATE** —
no invented stabilizers, no plausible corrections. An escalation after an exact match
that failed is a SUCCESS of the process.

## Standing mechanical preconditions (each earned by a real failure)
1. **fp64** — `precision_gate.require_fp64`. `JAX_ENABLE_X64=1` does NOT change the policy;
   `get_policy().storage` defaults to float32 and **constructors read it at build time**,
   so an fp64 state must be *built* under an fp64 policy.
2. **Time level** — `time_levels.time_level_for_dump` (fails closed). NEMO routines mix
   Kbb/Kmm/Kaa within one call; establish each input's level from source.
3. **`LEGOESM_NEMO_E3T` explicit** — the default `"off"` is a known-wrong 1-D ladder
   (12.9% off below k=25) and has contaminated four measurements.
4. **Metric identity** — before comparing two numbers from different scripts/ledgers,
   verify **same metric, same aggregation, same POPULATION, same reducer**.
   Four false conclusions today came from violating this.
5. **Dump provenance** — state which side of any later operation a dump sits on, with the
   F90 write line. Six confirmed cases where the dumped quantity was not the applied one.

## Recurring trap catalogue
- `avt_k`/`avm_k` are **closure-only**; EVD is applied to copies and never written back
  (`zdfphy.F90:313-323`). NEMO emits no post-EVD field.
- An MLF dump of `ts(Naa)`/`uu(Naa)` **after** a stage carries the whole step's accumulated RHS.
- A dump may predate a later overwrite (`ATF`: `mlf_baro_corr` rewrites `puu(Kaa)` *and*,
  under `ln_bt_fw=.false.`, `puu(Kmm)`, before `dyn_atf_qco` runs).
- A dump may omit an operation NEMO applies one line later (`eiv`: the minus at
  `ldftra.F90:833` vs the dump at `:864`; signature = corr exactly **-1.000000**).
- Filename tokens lie: `atf_dump_uu_before.bin` is **Kmm** (pre-filter *stage*, not Nbb).
- Probes go **config-blind**: `coverage_rows_measure.py` twice measured a static path
  regardless of the card. Fixing one blind dispatch does not fix its siblings.
- **Unit harness**: smooth **synthetic inputs can hide mask/boundary errors** (a mandatory
  `mask=` omission moved a synthetic case 0.9649→0.9673 but real data 0.706→0.9999999).
  Dynamic range and boundary coverage are different properties — cross-check geometry-
  sensitive routines on **real restart data**.
- Fortran **explicit-shape dummies** with decomposition macros (`A2D(0)`, extent 52 not 56)
  silently scramble (i,j,k) via sequence association; compiles and links clean.
- Halo/domain: some dumps are full `56x203` (halo=2), others interior `52x199` — use `_load_full`.
- legoESM raw `u`/`h_u` carry a **+1 column offset** vs NEMO's index (`_u_to_nemo` strips it).
- `w`/`G` live on the **interface** grid (37) vs `u`/`h_u` on cell centres (36).

## Harness artifacts found (rows that were never model defects)
`eiv transport u/v` (missing minus) · `dyn_ldf u/v` (probe fed NOW, NEMO reads Kbb) ·
`ATF filter u/v` (stale pre-`mlf_baro_corr` Kaa → **bit-exact**, now AT BAR) ·
`zdftke composite` (post-EVD reference) · `lbc_lnk sign` (no per-element entry) ·
`dyn_adv ZAD` **42.5% of the union metric** (dump carries below-seafloor Krhs that
`dynzdf.F90:121` discards). **The DEBT list overstates the model's real infidelity.**

## Live rows
- **`dyn_adv ZAD`** — highest leverage (dominates the v-unification, corr 0.9845).
  **ROOT CAUSE CONFIRMED**: `dynzad.F90:86` has no per-face `umask` guard, so NEMO averages
  `ww` from both T-neighbours at interface k+1 including one that is genuinely deeper/wet;
  legoESM's `face_active` min-rule mask (`vertical.py:1249-1252`) zeroes it. Explains
  **100.00%** of the active-only row (ratio 1.000, zero free parameters). **Real fidelity
  defect** — those u-faces are wet, so `dynzdf.F90:121` does not discard NEMO's result.
  active-only **3.0332e-02**, union 3.9993e-02. Fix in flight (gated option, default
  bit-identical, sign/conservation walk required).
- **`zu_frc` u (8.03e-3)** — drives the largest barotropic rows. **Nine candidates refuted.**
  Signature: broad envelope + damped ~34-row ripple, enhanced at the **periodic seam**
  (coincident with the sill). `zu_frc_write_ledger.py` (2026-07-30) enumerated EVERY
  `dynspg_ts.F90` write to `zu_frc`/`zv_frc` between :287-497 (self-check: table's write-line
  set == live-source regrep, exact match) and measured the two ACTIVE lines the six-piece
  budget omitted: `dyn_drg_init` (:381-382, dumped `drg_dump_zu_frc_inc.bin`) and the wind
  CENTRED term (:432). **Drag REFUTED as owner**: own-share 0.0001 (RMS 1.6e-10 vs zu_frc's
  3.0e-6), lego-vs-NEMO error RMS 1.56e-15 (machine-precision match, consistent with the
  already-recorded 0.999937/1.000278 gate row), corr with zu_frc's error -0.18/-0.15 (u/v).
  **Wind term UNMEASURED** — no NEMO-side increment-only dump exists (only raw `utau`); would
  need a new dump bracketing :420-433 the way :373-399 already brackets the drag call. On
  legoESM's side wind is structurally folded into the 3-D `du_dt` depth-mean (not a separate
  additive line, `surface_stress_implicit=False` on this card), so it isn't cleanly isolable
  without the NEMO-side dump either. Recommend **pause**: ledger now provably complete, drag
  ruled out, wind is the one remaining untested line but requires new instrumentation: out of
  scope for the unit harness (emergent solver behaviour).
- **`zdftke sh2`** — ESCALATION 1: exact transcription in, restricted-to-signal ratio 0.904.
  Family measured **climate-inert**, so parking is defensible.
- **`ldf_slp` ×4** — CONDITIONING-LIMITED, all three stopping-rule conditions verified.

## Measured strategic result — read before prioritising
The 5-year ACC acceptance run (protocol byte-identical, harness self-validated first) found
the campaign's **1e-6-class fixes are CLIMATE-INERT**: year-1 upper contrast 0.9082 → 0.9082,
ΔACC noise-level with two sign flips. The fixes change the state in the **wrong place**
(upper ocean, north of the channel band) while 80-98% of the missing thermal wind is sourced
**below 1000 m in the southern channel**. **⇒ Priority follows residual magnitude.** This does
not refute exactness-first — the 1e-2 rows remain untested — but it bounds the optimism.
Caveat: that run used `e3t=off` (the wrong 1-D ladder), matched to baseline for protocol
identity; a true-ladder acceptance is blocked on the restart-start instability.

## Human decisions — SETTLED 2026-07-30
1. `sh2` — **PARKED** (exact transcription in; family climate-inert).
2. Provenance — **re-measure only high-magnitude / fix-touched rows**; the machine check keeps
   the remaining 16 visible.
3. CONDITIONING-LIMITED — **counts as RESOLVED-WITH-NOTE** (the 4 `ldf_slp` rows).
4. `zu_frc` u — **PAUSED**, with one named resumption condition (below).
5. Push/PR — **done**: PR #1395, 91 commits.

---

# SESSION CLOSE 2026-07-30 — resume here

## Model health (verified, not assumed)
- **5-year from-rest DINO run completes clean** at the pushed HEAD with all six fixes active
  (ACC 55.578 → 65.540 Sv, census gates exact, ~11 min/yr). Card values were verified
  programmatically before launch, not assumed.
- **205 unit tests pass** across every touched module (`dino.py`, `tke.py`, `vertical.py`,
  `eos.py`, shortwave, box budget).
- **Non-NEMO recipes are byte-identical.** Every fix is a gated option, default = prior
  behaviour, each with a bit-identity test asserting `max|diff| == 0.0` (`==`, not `allclose`).
- **Known broken, and PRE-EXISTING**: restart-start on the true 3-D ladder
  (`LEGOESM_NEMO_E3T=both`) — max|u| 0.69 → ~3 m/s over 20 d. Predates this work (addendum 33).

## The strategic result — read this before spending anything
**TWO acceptance runs, both NULL.** Run 1: three 1e-6-class fixes. Run 2: the ZAD fix
(4.9e-3 → 3e-5, >100×, with a deep shelf-edge signature across 7255 columns). Year-5 ΔACC
**+0.0001 Sv**; upper contrast 0.8956 → 0.8954; the 5-year deep decay 0.4373 → 0.4370 untouched.
**Per-term transcription fidelity is not what costs the 25 Sv.**
**⚠ BUT BOTH RUNS USED THE WRONG LADDER** (`LEGOESM_NEMO_E3T` unset — 12.9% off below k=25),
kept for protocol identity with the baseline. **The ACC deficit is sourced below 1000 m, which
is exactly where that ladder is wrong.** So both nulls carry a background geometric error ~100×
the size of the fixes under test. **The nulls may be measuring the ladder, not the fixes** —
which is why the true-ladder instability is the campaign's critical path, not more gate rows.

## Live threads, with resume conditions
- **`zu_frc` u (8.03e-3)** — PAUSED. Nine candidates refuted. The **write ledger is provably
  complete** (`zu_frc_write_ledger.py`, self-check regreps live source): active writes are
  `:336` depth-mean, `:367` zu_trd subtraction, `:381-382` drag (measured, refuted),
  `:432` CENTRED wind. **RESUMPTION CONDITION NOW MET** — `wnd_dump_z{u,v}_frc_inc.bin` exists
  (batched rebuild `819ca1b59`). **One bounded comparison decides it**: lego's wind contribution
  to `F_slow_u` vs the dumped increment — RMS share, error corr vs the `zu_frc` error field,
  ripple/seam/asymmetry. Owns it → the largest row is solved. Doesn't → ledger exhausted, every
  line measured, close it as a bounded negative.
- **True-ladder instability** — the critical path. Eliminated by measurement: CFL, `ln_zad_Aimp`,
  `kappa_GM` magnitude, thin cells, slope cap, derived `gdept`, **GM-bolus discrete divergence**
  (2026-07-30: identical on both ladders, and structurally impossible — `nemo_eiv_bolus_transport`
  uses only `e2u`/`e1v`, no `e3` term), **abyssal slope-cap population** (flat, within ~10%).
  `use_gm_redi=False` (zeros κ_GM only, Redi untouched) still restores stability.
  **THE OPEN CONTRADICTION and the next cheap check**: κ_GM, slopes and bolus divergence are all
  ladder-insensitive, yet addendum 36 recorded the bolus entering the advecting flux **37% larger**
  on the true grid. Since `psi = κ × slope`, those cannot all hold. **Reconcile the 37%: is it
  real, and is it the same quantity?** (Five metric-identity incidents occurred on 2026-07-30 —
  treat any un-reconciled cross-script figure as suspect.)
- **zdftke composite** — real residual (corr 0.966; ~5% signal-weighted ratio), **INDEPENDENT of
  sh2** (substituting NEMO's own sh2 moved corr 0.9633 → 0.9656). Candidates, all now unblocked by
  the rebuild: buoyancy sink `p_avt*rn2` (`zdftke.F90:495`), `zmxlm` (`:814-819`), tridiagonal
  `zzd_up`/`zzd_lw` (`:485-492`).
- **MXL residual** — escalated; `tke_dump_rn2.bin` now exists. Re-walk with true `rn2`: expect
  0.0099 → ~1e-9 (confirming the rn2b-proxy explanation) or a real escalation.
- Also unblocked by `819ca1b59`: `wzv` row (`wzv_dump_ww_call1/2.bin` — **two calls per step**,
  use call2 for `tra_adv` consumers), `traldf_iso_lap` bracket, `dynzdf` stress bracket.

## Known harness flake — do not chase
Intermittent `buf_write` SIGSEGV in ~50% of bare NEMO runs, ASLR-sensitive, identical backtrace
in logs dated 2026-07-27. **Pre-existing, not ours.** Retry; registered dumps came from a clean run.

## Token economy (measured, in force)
Subagent lanes were **~95% of spend** (~30 lanes × 100–400k on 2026-07-30). In force: 3-hourly
cron (a FLOOR — fire manually for bursts), **ONE lane per iteration**, **explicit scope caps**
("do steps 1–2, report, STOP"), briefs point here instead of re-typing context.
**Protected: the adversarial reviews** (they caught a stale test encoding a bug as correct, a
wrong `Kmm` instruction, a missing mandatory `mask=`, a vacuous fp32 control) **and numeric
precision in any compression** — reconciliation needs old numbers exact and findable.
