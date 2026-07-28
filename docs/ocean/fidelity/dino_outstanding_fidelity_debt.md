# DINO / NEMO fidelity debt register (#1226)

Written 2026-07-27 in response to: *"how many more 'good enough's have you left
in there?"* — a full audit of every term where I declared PASS / MATCHED /
CLOSED / WAIVED while the measurement was not actually exact.

**The bar** (standing user directive): NEMO-faithful, corr 1.0 **and** ratio 1.0.
Not "close", not "within tolerance", not "attributed to X" on the strength of a
single black-box analysis.

**Method required to clear an item** (learned the hard way): dump the routine's
**internals** from NEMO, compare **stage by stage**, and name the first
deviating stage. Output-only comparisons hide compensating errors — that is how
`dyn_adv` passed for a day with a perfect KEG masking a broken ZAD, and how
"threshold chatter"/"irreducible amplification" got recorded as verdicts when
they were only ever hypotheses.

---

## A. Revoked acceptances (already reopened)

| # | term | measured | label I gave it | why it is debt |
|---|---|---|---|---|
| 8 | `traadv_fct` limiter | tendency 0.9923 | "FAITHFUL" | `nonosc` is deterministic; a true transcription must hit roundoff. Prime suspect for spurious diapycnal mixing. **IN PROGRESS** |
| 4 | `zdf_mxl` | 99.88% levels (12 cols) | "ACCEPTED" | "below input precision" never proven by comparing internals |
| 10 | `dyn_vor` EEN | 0.9999; bottom levels 1.04–1.07 | "ACCEPTED" | two fix attempts inert ⇒ mechanism still unknown |
| 7 | eiv transport | u 0.9985 / v 0.9954 | "near, accepted-for-now" | "diffuse, no lead" = not investigated to internals |
| 11 | `zdftke` composite | 0.9976 (257 cells) | "CLOSED" | "EVD threshold chatter" is a hypothesis, not a verdict |
| 5 | `ldf_slp` bottom row | \|x\| 1.0255 | "irreducible amplification" | asserted from a substitution test, not from internals |

## B. Called PASS/MATCHED with a ratio that is NOT 1 (never revoked — new debt)

| term | corr | ratio / error | what I said |
|---|---|---|---|
| `eos_rab` α | 1.000000 | median 4.7e-6 (β bit-exact) | "PASS" |
| `bn2` | 1.0000000000 | median \|rel\| 6.96e-6 | "PASS" |
| `ldf_eiv` κ | 1.000000 | \|x\| **1.000608** | "CLOSED" |
| `dyn_hpg` | 1.000000 | \|x\| **1.000045** | "MACHINE-EXACT" |
| `dyn_spg_ts` outputs | 0.9996–0.99999 | `puu_b` \|x\| **0.9871**, `un_adv` **1.0067** | "VERIFIED" — a 1.3% ratio gap, never explained |
| ATF filters | u 0.999969 | u \|x\| **0.9955** | "MATCHED" — 0.45% |
| `dyn_ldf` | 0.9979 / 0.9994 | \|x\| **1.0039** | "MATCHED" — 0.4% |
| `traadv_fct` fluxes | 0.99994 | \|x\| 1.0001 | "faithful" |
| slopes (interior) | ≥0.9988 | \|x\| 1.0011 | "CLOSED" |

## C. Never verified at all (waived, deferred, or structurally skipped)

| item | status | note |
|---|---|---|
| `mlf_baro_corr` | algebra-verified only | inlined at -O3; needs a `_step_impl` diagnostics hook. NEMO dumps 8883-8886 already exist |
| `dom_qco_r3c` **r3u/r3v** | never compared | `nemo_io` reader lacks `hu_0`/`hv_0`; only the T-point `r3t` was checked |
| `lbc_lnk` sign convention | deferred | needs a different harness |
| `zdf_mxl_turb` | UNVERIFIED | found by the runtime trace; turbocline depth, consumers unchecked |
| `zdf_drg_nonlin` + `dyn_drg_init` | "interface-covered" | bottom drag, never term-isolated |
| `dyn_cor_2d` (69×/step) | "interface-covered" | per-substep barotropic Coriolis, never term-isolated |
| tracer advection **salinity** | never compared | only `jp_tem` was dumped/compared; S assumed to follow |
| `traadv_fct` clean tendency | worked around | `trd` dumps are Krhs-contaminated; comparison used flux reconstruction, not a byte-level tendency |

## D. Unexplained anomalies (no owner, no hypothesis under test)

1. **legoESM recovers only 68% of its own ACC from its own density field**, while NEMO is thermal-wind self-consistent (1.46 by the same measure, i.e. consistent within the method's bias). A genuine momentum/reference-velocity residual, never chased.
2. **The 90-day twin's `u_surf` got WORSE** (0.9797 → 0.9691) after the momentum RHS was made machine-exact. Never explained; implies error downstream of the explicit RHS.
3. `dyn_spg_ts` `puu_b` \|x\| 0.987 — the entry-seed hypothesis was falsified (fix inert); no replacement hypothesis.
4. NEMO's `grid_T` history output is **numerically wrong** (votemper ×9.6 surface → ×290 deep; restarts are fine). Cause not diagnosed — an XIOS thickness-weighting/normalisation issue is suspected. Any past analysis reading NEMO T/S from `grid_T` is invalid.

## E. `ldftra` isoneutral diffusivity (Redi, `nn_aht_ijk_t=20`) — instrumented 2026-07-27

Investigated as the leading suspect for the 13%-too-weak upper-ocean meridional
T gradient (costs ACC via thermal wind). **Verdict: NOT the cause** — the
coefficient magnitude is bit-exact; a small (~2.6e-5) v-face discretization
residual was found and is now tracked, but it is far too small to explain 13%.

- **Formula correction (source read + live dump both confirm)**: NEMO's
  `nn_aht_ijk_t=20` (`ldftra.F90:290-296,322-326`, `ldfc1d_c2d.F90:106-155`)
  computes `ahtu/ahtv = (½·rn_Ud)·max(e1u,e2u)^1` — i.e. **`rn_Ld` is DEAD for
  this branch** (`aht0 = ½·rn_Ud·rn_Ld` is computed but never used by
  `CASE(20)`; only the printed `aht0` diagnostic references it). The task
  brief's "aht0 = 1350 m²/s (=½·rn_Ud·rn_Ld)" is a **documentation red
  herring** — NEMO's own `ocean.output` prints the value actually used:
  `"maximum reachable coefficient (at the Equator) = 1501.1854665553683 m2/s"`
  (`ldftra.F90:325`, `zah_max = zUfac*(ra*rad)^inn`). Confirmed independently
  by a live `kt==nit000` dump of `ahtu`/`ahtv` (MY_SRC/ldftra.F90, units
  8960-8963, `cfgs/DINO/RUN_1226_AHTU`).
- **legoESM's `K_h_base = 0.5*cfg.U_T*grid.radius*grid.dlon`** (dino.py:2520)
  evaluates to **1501.1854665553683** for both the default and
  `nemo_faithful_dino_config()` grids — bit-identical to NEMO's `zah_max`,
  because `grid.radius=constants.R_earth=6371229.0` and `grid.dlon≈1°` in
  radians reproduce `ra*rad*rn_e1_deg` exactly. The task's other stated
  number ("kappa_Redi = 1443.4475639955465") does not reproduce from current
  source with either grid path checked — likely stale/from a different
  config snapshot; the live value is 1501.1854665553683.
- **Field comparison (u-point, all 36 levels, both hemispheres)**: corr
  1.0000000000, ratio 1.0000000000, rel-err median 0, p90 1.7e-16 — roundoff.
  **AT BAR.** legoESM's `_static_kappa_redi_override` (T-point `cos(lat)`
  field) → `interp_cell_to_uface` (same-row average) is a no-op for a
  field that is constant along a row, so it reproduces NEMO's direct
  `cos(gphiu)` evaluation exactly.
- **v-point — FIXED 2026-07-27 (tier-2 item 1)**. The original note here
  ("`interp_cell_to_vface` averages `cos(φ_j)` and `cos(φ_j+1)`... `avg(cos) ≠
  cos(avg)`") was **WRONG** — `_static_kappa_redi_override` never called
  `interp_cell_to_vface`; grep + `git log -p --follow` confirm it has never
  existed in that function. That description was a probe-reimplementation
  artifact (Rule 0: "trace the real dispatch chain, a probe that re-implements
  a path is not evidence about that path"), not a description of the code that
  actually ran. The REAL cause: legoESM built ONE T-point field
  (`K_h_base*cos(lat_T)`) and reused it unshifted for BOTH the u-face and
  v-face Redi fluxes (`zfu`/`zfv` in
  `nemo_iso_lap_tracer_tendency_latlon_cgrid`, plus the shared
  `nemo_iso_w_kappa_sums`/`nemo_iso_a33` w-point kappa averages that feed the
  `ln_traldf_msc` implicit K33). NEMO's `nn_aht_ijk_t=20` does NOT do this:
  `ldf_c2d('TRA', ...)` (`ldfc1d_c2d.F90:141-145`) evaluates `ahtu` and `ahtv`
  as two INDEPENDENT arrays, `ahtu(ji,jj)=zUfac·MAX(e1u,e2u)^inn` and
  `ahtv(ji,jj)=zUfac·MAX(e1v,e2v)^inn` — i.e. `∝cos(lat_u)` and `∝cos(lat_v)`
  respectively, each at its own point. `cos(lat_u)==cos(lat_T)` exactly on
  this grid (u shares its T-row's latitude) so the u-face was already exact;
  `cos(lat_v)` is a genuinely different (row-shifted) value that legoESM was
  never computing at all.
  **Fix**: `_static_kappa_redi_override` now returns `(kappa_T, kappa_v)`,
  with `kappa_v = K_h_base·grid.cos_lat_v[1:]` (the true v-face latitude,
  north-face-of-cell-j convention matching `grid.dx_v[1:,:]`). A new
  keyword-only `kappa_Redi_v`/`kappa_redi_v_override` (default `None` ⇒ reuse
  the existing field, bit-identical for every other closure — Visbeck/EKE/
  GEOMETRIC/Treguier are genuinely T-point quantities that legitimately DO
  face-average the same way for u and v) threads this through
  `nemo_iso_lap_tracer_tendency_latlon_cgrid`, `nemo_iso_w_kappa_sums`,
  `nemo_iso_a33`, `gm_redi_tracer_tendency_latlon`, and
  `compute_isoneutral_K33_latlon`.
  **Verified against the actual NEMO dump** (`ldftra_dump_{ahtu,ahtv,gphiu,
  gphiv}.bin`, `RUN_1226_AHTU`, reproducible via
  `scripts/validate/ocean_fidelity/dino_1226/ldftra_ahtv_compare.py`):
  pre-fix v-face corr 0.9998618 / ratio 1.0000380 (NOT the previously
  recorded 1.0000260 — the earlier number came from whatever ad-hoc
  reimplementation produced the false "interp_cell_to_vface" story, not from
  this code); post-fix corr 1.0000000 / ratio 0.9999999704 — matches the
  u-face's own bit-exact quality. Ground-truth unit test:
  `tests/ocean/unit/test_dino_experiment.py::TestDinoLatLonModelConfig::test_static_kappa_override_row_scaling`.
- **Residual open question**: the ahtu/ahtv coefficient is now the LEAST
  likely explanation for the 13% front deficit found so far. The search for
  that deficit should move to a different term (candidates from this same
  ledger: `traadv_fct` vertical upstream flux over-clip 1.7-1.9x — row
  "traadv_fct VERTICAL upstream flux" above, added by a parallel
  investigation — or the slope/taper chain `ldf_slp`).

---

## Honest count

- **6** acceptances revoked
- **9** terms labelled PASS/MATCHED with a non-unit ratio
- **8** never verified at all
- **4** unexplained anomalies

The step-by-step methodology was created precisely to prevent this, and then I
applied it with a tolerance it does not have. Nothing in A–C is known to be
wrong; the point is that **none of it is known to be right** at the stated bar,
and the campaign has now produced enough retractions to make that distinction
load-bearing.


## F. RETRACTED findings — read this before re-investigating

Retractions must live where the next session looks, or they get re-discovered
at full cost. This section is that place.

| retracted claim | why it died | date |
|---|---|---|
| "Tracer operator COMPOSITION differs from NEMO (GM/Redi mutates the field advection reads)" | TRUE for `nemo_dino_kamm` (forward_euler) but **NOT for `nemo_dino_kamm_mlf`**, the recipe under test: `_leapfrog_step` passes hardcoded `_ab2_scope_override="advective"` (ocean_model_latlon_cgrid.py:6836,6868), routing GM/Redi to the ADDITIVE `_diss_dT_incr` bucket; the mutation branch (:3873) is dead code there. Final update :6925 is additive. Only the single first from-rest step mutates (1 of 58,400). The probe hand-reconstructed the mutating path and mislabelled it "actual code order" without tracing dispatch. NOTE: an earlier session had already retracted this in `dino_session_2026_07_25.md` — it was re-discovered because that retraction was not carried forward. | 2026-07-27 |
| "N² is 54% too large" | interface index off-by-one in the comparison; correctly aligned = corr 1.000000 | 2026-07-26 |
| "hmlp is 14% too deep" | unvalidated depth lookup; control test now reproduces NEMO's hmlp to 0.0 m | 2026-07-27 |
| "slopes are 9% too large" | signed-sum ratio on a SIGN-CHANGING field inflated a 1.2% magnitude bias ~9x | 2026-07-27 |
| "wslpi corr 0.9585" | alignment artifact; a proper ±2 offset scan peaks at 0.999110 | 2026-07-27 |
| "col_stretch recovers (1+r3t)" | `h_partial` is the STATIC at-rest thickness; measured identically 1.0 | 2026-07-27 |
| "dry-vertex e3f_0 fallback explains the EEN bottom bias" | controlled A/B was byte-identical | 2026-07-27 |
| "the vertical upstream flux is corr 0.91" | OFFSET ARTIFACT — the probe headline scored `offset=+1`; at the correct `offset=0` the advecting transport, upstream flux and upstream tendency all match at corr >= 0.9977. (A REAL but climate-inert dry-cell masking bug was found while chasing it: w-face clips 1126-1256 -> 603 vs NEMO 662, commit 01c1f226a.) | 2026-07-27 |
| "the harness biases EVERY measurement ~0.44% (e3t default) + 30-60 ppm (dy_v vs e2v)" | MOSTLY REFUTED by measurement. The e3t default affected ONLY `eiv transport`; dyn_ldf/ATF/dyn_spg_ts/zdftke/ldf_slp/ZAD/salinity are identical under off vs both. The dy_v/e2v part is refuted outright: on non-tripolar grids `gradient_*_cgrid`/`divergence_cgrid` recompute face metrics inline and NEVER read grid.dx_u/dy_u/dx_v/dy_v, so substituting NEMO's metrics is a structural no-op. | 2026-07-27 |
| "dyn_hpg is AT BAR (1.000000008)" | Measured on a DEGENERATE from-rest state (ssh=0, zonally-uniform IC) where `du` is identically zero on both sides and the zuap/stretch terms vanish. On the actual Y5 twin state it stays 1.000045. The number certified the trapezoid p'/EOS/g/gradient path only. | 2026-07-27 |
| "ldf_eiv kappa has an INDEPENDENT operator defect (corr 0.975)" | MEASUREMENT ARTIFACT. The probe compared lego's raw T-POINT kappa against NEMO's `paeiu`, which is the U-FACE AVERAGE (ldftra.F90:741, 0.5*(zaeiw(i)+zaeiw(i+1))*ssumask) -- a different quantity; the probe's own comment wrongly asserted they were the same. Routed through the existing `nemo_kappa_gm_to_faces`: corr 0.975163 -> 0.999995, ratio 1.032510 -> 0.999958. Every named intermediate (zn/zah/zhw/zRo/zaeiw) was ALREADY at 0.99999-class -- there was no deviating stage. NOTE the 3x3 alignment scan still picked offset (0,0) because both fields are smooth: **a passing alignment scan does NOT prove you are comparing the right QUANTITY.** | 2026-07-28 |
| "the FCT limiter is the climate lever" | centered (unlimited) advection reproduces fct2 to 4 decimals over 5 years; ACC identical | 2026-07-27 |

**Five probe/relay retractions in one day** (composition reconstruction,
vertical-flux offset). EVERY headline number must carry its own offset/alignment
scan BEFORE it is reported as a finding — an agent reporting a bare correlation
without one is reporting an unverified quantity.

**Rule 0 applies to OUR code too.** A probe that re-implements a code path is
not evidence about that code path — trace the dispatch chain.
