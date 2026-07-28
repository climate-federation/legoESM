# DINO / NEMO fidelity — human-readable status board (#1226)

Snapshot: 2026-07-28. Machine-checked counts come from
`scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py`
(**AT BAR 10 · DEBT 29 · UNMEASURED 6 · total 45**).

**The bar**: corr = 1.0 AND ratio = 1.0. Anything less is DEBT, whatever prose
was written about it. A term is cleared only by comparing the routine's
*internals* stage by stage — never by explaining its residual away.

---

## 1. FIXED — real bugs found and corrected

Each was a genuine transcription defect, found by internals comparison, each
with a test and a measured before/after.

| # | Bug | Effect |
|---|---|---|
| 1 | `zdf_mxl` missing NEMO's `MIN(jk,mbkt)` bottom cap | 339 columns ran past their own seafloor; 353 → 14 mismatched |
| 2 | α/β interpolated at the gdept midpoint instead of the true `gdepw` (`eosbn2:1459`) | MLD integrand 3.66e-4 → 4.95e-5 |
| 3 | Slopes fed a parcel-displacement N² instead of NEMO's `rn2b` | depth-growing bias; \|wslpi\| deep 1.0148 → 0.9995 |
| 4 | **Wrong gravity on the oracle card** (canonical Earth vs NEMO standard) | 5.0e-5 in *every* buoyancy term |
| 5 | `bn2` divided by reference `e3w` instead of live | partially cancelled #4, which is why neither showed alone |
| 6 | κ_GM silently ran a *different* N² than its own slopes (getattr fallback) | the whole 1.6% aeiu deficit |
| 7 | **Shapiro smoother treated the periodic seam as a wall** | seam slopes 0.75×, squaring into a 0.55 `zah` deficit |
| 8 | Missing NEMO per-face κ averaging in the bolus ψ | eiv transport rel-err 3.4% → 0.16% |
| 9 | TKE Prandtl `zri` off by a factor K_M (numerator kept, denominator weighting dropped) | Pr clamped to 1 where NEMO saturates at 10 |
| 10 | **`dynzad` computed the FLUX form `∂(wu)/∂z`; NEMO uses ADVECTIVE `w·∂u/∂z`** | ZAD corr 0.62/0.35 → 0.9992/0.9987 |
| 11 | `nonosc` bound omitted the upstream guess (`max(pbef,paft)`) | 40–45% of wet cells; horizontal tendency → 0.99995 |
| 12 | Dry cells unmasked in the Zalesak bounds (garbage ~1e17 leaked into wet neighbours) | w-face clips 1126 → 603 (NEMO 662) |
| 13 | `eos_depth="geometric"` on the card was **hardcoded to `insitu`** in the density builder | eiv v 0.982 → 0.991 |
| 14 | Ω was a 4-significant-figure literal, 1.59e-5 low — and it is *squared* into κ_GM | 3.16e-5, proven by substitution collapsing to ≤9e-16 |
| 15 | `ahtv` built from one T-point field for both faces; NEMO builds `ahtu`/`ahtv` independently | ahtv → bit-exact, AT BAR |
| 16 | Live per-column `gdept` reached only 1 of 4 consumers; PGF/slopes used the static ladder | `dyn_hpg` du → **AT BAR**, accumulation profile gone |
| 17 | The `(1+r3t)` Jacobian applied unconditionally (valid only for partial-cell coords) | latent NaN, caught by a Treguier dispatch test |

**Terms now AT BAR (10):** `sbc`, `eos_rab` β and α, `bn2`, `ldftra ahtu`,
`ldftra ahtv`, `dyn_adv` KEG, `zdf_drg_nonlin` coefficient, `dyn_hpg` du,
`ssh_nxt` (corr).

## 2. IN PROGRESS — running right now

| Work | Why | Status |
|---|---|---|
| **`ldf_slp` internals stage audit** | 5 rows depend on it (4 slope components + κ_GM downstream); every cheap cause already exonerated | already moved `wslpi` 0.997860 → **0.999963** via an `active_3d` mask fix |
| **Board re-measurement for provenance** | 37 of 45 rows had no measured-at commit; staleness already caused a false regression alarm | in flight |

## 3. TO FIX — the remaining debt, grouped by what it needs

### (a) One unidentified defect, 5 rows — highest value
`ldf_slp` wslpi / wslpj / uslp / vslp, and `ldf_eiv` κ (which *cannot* move
until the slopes do). Exonerated by measurement already: live depth (r3t ~7e-5,
two orders too small), metric convention (moved 0.0), the diffusivity
coefficient (bit-exact), N² and α (both AT BAR), the seam.

### (b) Salinity — the most alarming single row
`traadv_fct` SALINITY: corr **0.203** / ratio 3.17. Not a boundary artifact
(refuted by measurement). Mechanism: horizontal and vertical tendencies each
match at 0.9999 but **cancel to 1.2% of gross**, so each component's residual is
~5× the net signal. Salinity carries ~35% of DINO's density span, so this feeds
the EOS and the ACC. *This is the quantitative proof the exactness bar is
necessary: 0.9999 per component gives 0.20 on the applied tendency.*

### (c) Blocked on the deferred v-face metric
`dyn_hpg` dv (substituted → 1.000000007), and probably part of `dyn_cor_2d` v.
Blocked because `vface_zonal_cos_lat` is a tested #516 invariant
(strain/stress adjointness + div/advection mass consistency) that must not be
weakened to move a fidelity number.

### (d) Unexplained residuals, cause not identified
`dyn_spg_ts` puu_b (1.3%, the largest unexplained), un_adv, pssh;
`ATF filter` u (0.45%); `dyn_ldf` u/v; `dyn_vor` EEN u/v (bottom levels);
`zdftke` pdlr and composite; `traadv_fct` fluxes / tendency / vertical flux;
`dyn_drg_init` (u/v asymmetry larger than any known mechanism predicts);
`dom_qco_r3c` r3t/r3u/r3v; `ssh_nxt` ratio.

### (e) Never measured
`zdf_mxl` nmln (12 knife-edge columns), `mlf_baro_corr` (needs a `_step_impl`
diagnostics hook — NEMO dumps already exist), `lbc_lnk` sign convention,
`traadv_fct` horizontal tendency ratio.

### (f) Known missing / structural
- **`zdf_mxl_turb`** — no legoESM equivalent. NEMO's `hmld` turbocline is
  diagnostic-only and never feeds dynamics, so this is waivable, but it is a
  genuine missing term.
- **legoESM is UNSTABLE on NEMO's true vertical grid.** `LEGOESM_NEMO_E3T`
  defaults to the *wrong* 1-D ladder because `both` blows runs up
  (max\|u\| 0.66 → 3 m/s). That default has contaminated three measurements.
  This is arguably the most important open defect: we cannot run faithfully on
  the oracle's own geometry.

## 4. THE CLIMATE QUESTION — still open, honestly

From rest, 5 years: legoESM ACC **67.6 Sv** vs NEMO **91.1 Sv** (0.74×). The gap
opens roughly linearly, i.e. a persistent *rate* deficit.

**Not chaos** — ensemble spreads are 0.15 Sv (legoESM) and 0.24 Sv (NEMO), so
the gap is ~100× the combined noise. Both models are near-deterministic at 1°
with parameterized eddies.

**Two distinct deficits, separated in time:**
- Upper ocean (<1400 m): meridional ΔT ~13% weak, flat from year 1.
- Deep (≥1400 m): matched at year 1 (0.95), collapses in year 2 (0.69) and
  year 3 (0.40). NEMO *builds* deep contrast; legoESM stalls and reverses —
  failing to build it, not losing it.

**Exonerated by measurement (8):** tracer advection scheme (centered vs FCT
identical to 4 decimals over 5 years), Redi coefficient, deep K_v, GM bolus
(timescale 64–643 yr vs a 5-yr spin-up), ZAD momentum (fix was climate-inert),
surface forcing + initial condition, tracer operator composition, vertical
transport.

## 5. METHOD — what this campaign learned the hard way

1. **Coverage, not a checklist.** The 11-item sweep verified suspected terms
   while the barotropic solver sat unenumerated. Now: every `CALL` in
   `stpmlf.F90` has a written disposition (`dino_step_chain_coverage.md`).
2. **Internals, not endpoints.** A perfect KEG masked a completely wrong ZAD
   for a day. Input/output comparison cannot see composition or ordering.
3. **The bar lives in a gate, not in judgment.** 0.99x recorded as "matched"
   repeatedly, across a whole campaign, while being told the bar each time.
4. **Every headline needs an alignment scan and stated provenance** (state +
   e3t mode + measuring commit). Multiple "findings" were offset artifacts.
5. **Rule 0 applies to our own code.** A probe that re-implements a code path is
   not evidence about that path — trace the dispatch.
6. **Matching an oracle sometimes means reproducing its approximations.** NEMO's
   isotropic Mercator metric is an approximation; legoESM's exact spherical-cap
   integral is *more* accurate and therefore *less* faithful.
7. **Cancellation sets the required accuracy.** Components at 0.9999 that cancel
   to 1.2% of gross give an applied tendency at corr 0.20.

Retractions are logged in `dino_outstanding_fidelity_debt.md` §F — read it
before re-investigating anything.
