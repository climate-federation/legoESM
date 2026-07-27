# DINO NEMO-oracle — program handoff (2026-07-24)

*Machine- and person-agnostic status record. Everything here is committed to the repo
so a fresh checkout on any machine reconstructs the program state without local memory.
Branch `feat/nemo-dino-topo-bridge` → PR #1313. Living issues: #1226 (fidelity gap),
#1317 (zdftke). Supersedes the pre-halo-strip `dino_program_summary.md` (kept as history).*

## Goal
A verifiable, differentiable NEMO oracle in legoESM: match NEMO 5.0.2's DINO experiment
(Kamm et al. 2025, 1° "R1") **term by term, exactly** — transcribe from NEMO source, never
diagnose from first principles or add stabilizers NEMO lacks. NEMO artifacts live under
`~/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/` on the dev box; the harness is
`scripts/validate/ocean_fidelity/dino_1226/`.

## Method (the discipline that repeatedly worked)
- **Transcribe, don't paraphrase.** Read the NEMO source line; build the exact form as a
  selectable option in the canonical module; default byte-identical; prove it with a
  NEMO-derived test. Every "fast" build that skipped this shipped a bug (sign flip,
  factor-2, dtype crash — all caught by tests/review, each costing a re-do).
- **Measurement-verified attribution only.** Read-based localization failed repeatedly;
  only controlled measurement survived. Change ONE variable, hold the eval protocol fixed.
- **Swap-the-subsystem discriminators.** To decide whether a residual is owned by subsystem
  X, replace X with a zero-transcription-risk version on BOTH models and see if the residual
  survives. Used decisively for the SST bias (constant-coefficient mixing, below).
- **Alignment tables.** For a whole subsystem, build the ordered N1..Nk step-for-step table
  (NEMO file:line vs lego file:line, MATCH/DIFF) before touching code. Closed the barotropic
  solver and drove the zdftke work.
- **Twin protocol.** `kamm_twin_90d.py`: initialize lego from NEMO's developed day-180
  restart (`--bridge-tke` + `--bridge-before` bridge all prognostic + integrator-memory
  state, so day-0 lego == NEMO bit-exactly), run both 90 days from the identical state,
  compare. HARD day-0 gate makes a rest-state "twin" impossible (see Traps).

## Current state (2026-07-24)

**PR #1313 — 28 commits, review-ready, not merged.** Contains the whole program below.
Each substantive commit had an adversarial (physics-validator) review; findings fixed
in-branch.

**Issue #1226 (fidelity gap) — close-ready after merge.** Its three title complaints:
(1) isoneutral mixing runaway — RESOLVED (upstream #1233 + faithful namtra_ldf re-enabled);
(2) smoothness 0.41× — RESOLVED (halo-strip fix → SSH small-scale ratio 1.00);
(3) ACC 18-vs-69 Sv — pathology RESOLVED (decay cured), residual re-scoped (below).

**Issue #1317 (zdftke) — closure faithful; SST bias re-attributed to dynamics.**

### What was fixed, in order
1. **Halo-strip root cause** (`8e5cb12ca`): NEMO 5 files are haloless; read with `nn_hls=0`
   gives the TRUE 199×52 domain with the real continent. Cell census exact (342,134 wet +
   face masks). Fixed the smoothness deficit outright; every prior "wall anomaly" was a
   subdomain-frame artifact.
2. **Sub-seafloor tracer leak** (`0db4c1abd`): `pn2` masked at construction (eosbn2.F90:1467)
   + 3-D MLD masks. Poison gate landed (`poison_gate.py`): dry-cell values → exactly 0 wet
   response.
3. **Barotropic solver — verified NEMO-identical.** Full N1–N16 alignment vs `dynspg_ts.F90`
   DINO-active branch (nn_bt_flt=2 centred boxcar). One real gap found+fixed: mid-step
   AB3-extrapolated continuity-flux depth (`3c3e85792`), with a per-substep NEMO-sequence
   oracle test. Cards flipped to the fully NEMO-true composition (drag triple ON, `alpha=0`
   crutch removed, nemo_ssh_avg face depths, centred forcing, Kmm EEN seed, dyn_drg).
4. **zdftke closure** (`e1744a053`..`c8cdb245d`): Phase-1 30-row alignment table (on #1317)
   → Phase-2 + gap-closure implementing every DIFF/PARTIAL row (T3 exact surface avm, T4
   Burchard shear, T6 explicit buoyancy sink, T8/T13 rn2b operands, T15 per-column bottom,
   T21 ceiling off, T23 max-floor backgrounds, T5/T25). Three latent bugs found while
   tightening (T3 double-count ~2× inflation; before-velocity staggering flag; Mode-B
   missing rn2b threading).

### Year-5 certification (zero-deviation card, from-rest, matched protocol vs NEMO RUN_5Y)
| metric | prior-era (superseded) | zero-deviation card | NEMO y5 |
|---|---|---|---|
| ACC | 16.8 Sv, **decaying** | **65.6 Sv, held flat** | 91.1 |
| SST corr / bias | 0.991 | 0.995 / −0.17 K | — |
| SSH small-scale ratio | 1.00→0.82 | 0.86 | 1.00 |

The **ACC decay pathology is cured** (was actively destroying transport; now equilibrates
and holds). Residual = a static 0.72× deficit, re-scoped below.

### NEMO is mid-spin-up at year 5 (important for reading every ACC number)
Kamm 2025 quotes equilibrium ACC = **206 Sv (R1)**. A NEMO 20-year run (`RUN_20Y`, yearly
dumps, banked for future lego long-runs) shows ACC climbing 60→91→...→**145 Sv at year 20,
still +1.8 Sv/yr, no plateau** — the year-5 91 Sv is a spin-up transient toward ~206. So the
0.72× is a comparison of two *slopes*, not two destinations; the meaningful ACC test is the
eventual 20-year lego-vs-NEMO trajectory pair.

## Open questions + ranked next steps

**The SST cool bias / thermocline residual is NOT vertical-mixing-owned.** Constant-coefficient
discriminator (`ln_zdfcst` both models, matched EVD, zero transcription risk): the −0.88 K
lego−NEMO bias and its drift-profile shape SURVIVE the closure swap intact (slightly
amplified). TKE is exonerated as the primary driver. The bias lives in the **dynamics** —
consistent with the ACC decomposition (lego's Drake-gap shear tracks its OWN density
faithfully → the thermocline slope is set by advection/PGF/GM, not mixing).

**Next discriminators (same swap-the-subsystem method), ranked:**
1. **GM/Redi isopycnal heat transport** — leading suspect for thermocline-slope / poleward
   heat differences.
2. **Momentum/tracer advection at the ACC band** — the 46% baroclinic-shear share of the ACC
   deficit lives here.
3. **PGF** — the 54% barotropic-offset share (bottom flow weaker in the gap → form-stress /
   bottom-drag balance at the sill).

**Deferred (documented, not built):**
- **T11 (zdftke coefficient vintage)** — NOT built. The discriminator proved the bias isn't
  TKE-owned, so the ~45-site state-threading spend isn't justified. Mapped in #1317 if ever
  needed.
- T4 staggered spatial stencil, FE-card before-state ceiling, T25 rn2b-in-EVD-trigger —
  structural/minor faithfulness residuals, documented in the tke.py docstrings.
- East-wall checkerboard: RESOLVED as a from-rest spin-up feature (no regrowth from a
  developed state; lego damps it ~2× slower than NEMO). Not an instability.
- Asselin dry-cell masking test failure (pre-existing) — triage queued.

## Instrument inventory (committed, reusable)
Under `scripts/validate/ocean_fidelity/dino_1226/`:
- `kamm_twin_90d.py` — canonical state-initialized twin (day-0 gate, `--bridge-tke`,
  `--bridge-before`, `--vmix-scheme`, `--save-3d`).
- `heat_discriminator.py` — redistribution-vs-loss vertical-drift discriminator.
- `mode_projection.py` — checkerboard-mode projector (reuse, don't re-derive).
- `poison_gate.py`, `compare_fullframe.py`, `dino_year_screen_fullframe.py`, budgets.
Session scratchpad npz/PNG artifacts are LOCAL and regenerable from these scripts + the NEMO
runs (`RUN_1Y`/`RUN_5Y`/`RUN_20Y`/`RUN_90D_TWIN`/`RUN_STEPDUMP`).

## Traps learned (each cost real time; each is now a rule)
- **Labels lie, fields don't.** Three instrument-class traps, all found by measurement/field
  inspection, none by reading comments: (1) KEG-bundling artifact; (2) `vtrd_zdf` is ~98.6%
  barotropic-strip integrator bookkeeping, NOT friction — **never compare lego tendencies to
  raw NEMO `trddyn` terms without subtracting the bookkeeping**; (3) a "twin" runner that
  silently ran from rest on NEMO's topography (day-0 gate now prevents it).
- **Matched windows / matched protocol.** Compare from-rest-vs-from-rest, developed-vs-
  developed, same averaging window. A number computed on a different sampling is a confound.
- **Faithful-but-worse is a signal, not a reason to revert.** Removing compensating errors
  (e.g. additive mixing backgrounds) can worsen a metric while being correct; the residual it
  exposes is the real target.
- **Mixed precision.** The model runs float32 fields under x64; tests in pure x64 can miss a
  dtype-carry crash. The shared tridiag solver now unifies input dtypes at entry.

## Program provenance
NEMO 5.0.2 built locally (env `nemo-build`, native iom_nf90, no XIOS). Reference PDF:
`docs/references/Kamm_etal2025.pdf` (not committed — cite by name). Full comment-by-comment
narrative on issues #1226 and #1317.
