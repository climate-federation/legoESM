# FV3 single-implementation program (Phases 1–3)

User directive (2026-07-10): "we should only have a single FV3 implementation
not many" + "fix FB's edge mode first, then make it the one." Phase 0 (the FB
covariant-convention fix) is DONE — PR #967, codex APPROVE, 120-day modon
clean. This document is the resumable execution plan for the remaining
phases. Update the checkboxes as milestones land; each milestone is one
PR-sized unit with its own codex review.

## Ground truth (Phase-0 outcomes this program builds on)

- The FB chain (`fv3_fb_sw_step`, duogrid-only) is the algorithmically
  faithful FV3 SW core: c_sw → p_grad_c → d_sw1..6, covariant winds
  internally (entry/exit conversion `fb_v_d_to_covariant/orthogonal`).
- Validated FB damping preset for vortex cases:
  `nord=1, d4_bg=0.16, dddmp=0.2, damp_v=0.02, nord_v=2` (120-day modon
  clean, mass 1e-15). W2-class steady flows carry a ~1.3 m/s/day linear
  drift (open follow-up, not an instability).
  *(SUPERSEDED 2026-07-12, M1b increment 2b: the "not an instability"
  part is FALSE — W2 additionally excites an exponential seam mode that
  NaNs C36 at day 7.5; the 5-day matrix window merely ends before C36's
  blowup.  See M1b item 2c under item 2.)*
- Production paths (`fv3_sw_tendencies` + RK3 in `FV3EdgeShallowWaterModel`,
  `CDGridShallowWaterModel`) are centered/calibrated research schemes —
  bit-identical through Phase 0, to be RETIRED in Phase 1.
- Known convention defect in production `d2a2c_vect` (orthogonal winds through
  covariant formulas) — prime suspect for the 0.344 m/s production W2 cube
  v-imprint. Fixing it is part of Phase-1 M4 (the retirement makes it moot on
  the SW side; the PE consumer is Phase 2).

## Phase 1 — FB becomes THE cube SW implementation

- [x] **M1 — FB production readiness.** DONE 2026-07-12 — deliverables
  landed (bench + calibration + drift attribution + matrix FB lane), but
  the findings BLOCK the M2 flip as things stand.  See "Phase-1 M1
  results" below.  Exit satisfied via documented waivers (W2/W5/W6).
## Phase-1 M1 results (2026-07-12)

All runs: C36, dt=300 s, x64 CPU, matrix IC recipes + matrix metric
pipeline held fixed; FB lane = duogrid (FB is duogrid-only) + the M1
preset `nord=1 d4_bg=0.16 dddmp=0.2 damp_v=0.02 nord_v=2`
(`fb_m1_preset_config()`).  Production baseline =
`results/atm_matrix_sw_full_cube/summary.json`; FB run =
`results/atm_matrix_sw_fb_cube/summary.json` (regenerate with
`scripts/matrix/run_atmosphere_test_matrix.py --only sw --grid
cubed_sphere --sw-core fb`).

**1. Performance (scripts/bench/bench_fv3_sw_fb_vs_production.py, AOT,
steady-state per-step ms; production = the EXACT matrix W2 lane config
incl. 2x biharmonic hyperdiff, codex M1 fix):**

| res | production | production-on-duogrid | FB | FB/prod |
|-----|-----------|----------------------|----|---------|
| C36 | 1.33 | 1.66 | 4.43 | **3.33x** |
| C48 | 2.50 | 2.40 | 6.54 | **2.62x** |
| C96 | 5.45 | 10.69 | 23.23 | **4.27x** |

FB is ~2.6–4.3x slower per step at the SAME dt (run-to-run thread noise
~±20% on this host; an earlier round without the production hyperdiff
read 4.1–5.2x).  Part is the duogrid halo (production_duo column); the
bulk is the d_sw chain — separately-jitted stage attribution at C36:
d_sw1..6 = 3.34 ms (d_sw3 B-grid KE PPM transport alone 1.67 ms), c_sw
1.41 ms, entry+exit covariant conversion 0.60 ms, p_grad_c + phase-4
PGF 0.28 ms.  FB compile is also ~10x production (28–30 s vs 3–5 s).

**2. Williamson-suite calibration (FB M1 preset vs production matrix):**

| case | production | FB (M1 preset) | verdict |
|------|-----------|----------------|---------|
| W2 5d | L2=4.68e-04, v_ll_Linf=0.541 | L2=1.54e-02, v_ll_Linf=43.8 | **FAIL** (33x / 81x worse) |
| W5 15d | PASS, mass 1.6e-16 | **BLOWUP day 12.5** (metric 1215) | **FAIL** |
| W6 14d | PASS, mass 0 | **BLOWUP day 9.0** (NaN) | **FAIL** |
| cosine_bell 12d | L2=1.89e-01 | identical | N/A (core-independent: CB never calls model.step) |
| cosine_bell_a0 12d | L2=1.91e-01 | identical | N/A (same) |
| colliding_modons 100d | PASS, mass 0 | PASS, mass 3.6e-16 | **OK** (+ Phase-0 120d C48 clean) |

Tuning within Fortran-plausible ranges CANNOT rescue W2/W5/W6:
- W2 worsens MONOTONICALLY with more damping (damp_v 0.02→0.12:
  v_north_Linf 55.6→206 m/s; d4_bg=0.25 blows up at step ~250).  Best
  found: `d4_bg=0.12, damp_v=0` → v_north_Linf 36.5 m/s, L2 1.49e-02 —
  still ~30x/60x off production.
- W5/W6 blow up (day 8.3–12.7) at ALL of damp_v∈{0.02,0.06,0.12} ×
  d4_bg∈{0.12,0.16} × dddmp∈{0.2,0.5} × nord∈{1,2}.  Production needed
  2x biharmonic hyperdiff for these long wave cases; the FB chain has no
  equivalent scale-selective enstrophy sink that its d_sw5/d_sw6 knobs
  can emulate at C36.
- WAIVER: the M1 preset is validated for the vortex family (modons)
  ONLY.  No FB preset family exists for W2-class steady flows or
  W5/W6-class propagating waves.

**3. W2 linear-drift attribution (scripts/tmp/_fb_w2_drift_ab.py, W2
C36 2d, raw phase chain, max|v−v_ref| drift):**
- base (per-step convert, d_sw3 overrides ON): 3.34 m/s/day (seam rows).
- (a) d_sw3 one-sided overrides OFF: 3.42 m/s/day → NOT the source
  (+2%, noise).
- (b) covariant-space run (convert IC once, NO per-step exit/entry):
  5.39 m/s/day → the exit-conversion 2-pass approximate inverse is a
  seam FILTER that REDUCES the drift by ~2 m/s/day, not the source
  (consistent with the fb_v_d_to_orthogonal docstring: exact inverse
  measured worse).
- (c) dt=150/300/600: 3.61/3.34/3.32 m/s/day — drift/day is
  dt-INDEPENDENT ⇒ the source is a STEADY spatial-truncation tendency
  imbalance (per-unit-time forcing), not an accumulating per-step kick.
  A per-step source would scale 2:1:0.5.
- Conclusion: the drift is the covariant chain's seam-row truncation
  residual on steady zonal flow (the single-step A+B imbalance measured
  by scripts/tmp/_fb_stage_decomp.py), pumped continuously; the del-6
  vorticity damping AMPLIFIES the resulting seam mode.  No small
  faithful fix exists (Phase-0 already falsified the exact-inverse and
  cross_face_halo candidates); the fix is a Phase-1 work item on the
  cross-face halo / corner-metric consistency of c_sw+d_sw at seam rows.

**4. Matrix FB lane:** `--sw-core {production,fb}` (default production)
in `scripts/matrix/run_atmosphere_test_matrix.py`; argparse choices +
defensive raise; CLI round-trip test
`tests/unit/test_matrix_sw_core_cli.py`.  Cube SW cases only.

**M2 GATE: DO NOT FLIP.**  FB currently loses to production on speed
(4–5x) and on every non-vortex SW case.  Before M2: (i) fix the seam
truncation forcing (item 3), (ii) give the FB chain a working
long-wave stabilization path, (iii) revisit step cost (d_sw3 PPM + c_sw
dominate).
*(Update 2026-07-12, M1b item 2: (i) substantially advanced — the seam
metric fix moved W5/W6 from mid-run blowup to passing the matrix gates
with the unchanged M1 preset, no new stabilization path.  Still short
of the gate: W5 winds still grow late in the window (see item-2 honesty
caveat), W2 steady-flow accuracy (v_ll_Linf 26.2 vs production 0.541 —
the first-order smooth vorticity-flux imbalance follow-up), and (iii)
step cost.)*
*(Update 2026-07-12b, item 2b below: the W2 residual is CLASSIFIED —
verdict (b), genuinely sub-oracle.  Published Fortran-FV3 W2 levels
(Mouallem et al. 2023, JAMES, 10.1029/2023MS003712) put the duo-grid
oracle at h-l2 ≈ 6e-5 / v-err ≈ 0.05 m/s at C48 5 d with clean
2nd-order convergence; production (4.68e-4 / 0.54 m/s at C36) sits at
the published KINKED-grid level, so the parity gate is CORRECTLY
referenced — if anything the bar for a duogrid FB chain is the
duo-grid numbers, i.e. HIGHER than production.  FB-W2 is ~2 orders
above the oracle and ANTI-convergent: C48 erupts at the equatorial
seams ~day 4.5 (both dt=300 and fixed-Courant dt=225), C72 NaNs
~day 3.5.  This further ADVANCES THE FALL-BACK CASE for the flip.)*

- [x] **M1b — fix-forward (USER DECISION 2026-07-12, time-boxed) — DONE
  2026-07-13; all items landed, W2 parity NOT reached -> FALL BACK
  (see "M1b FINAL VERDICT" below).**
  Three work items, in order; if all three land without reaching
  W2/W5/W6 parity, FALL BACK to consolidating onto the production core:
  1. [x] d_sw5 nord>=2 wider-halo port (concrete gap: the opt-in raises on
     nord>=2; Fortran stabilizes the W5/W6 wave class with nord=2-3
     del-6/del-8 divergence damping — sw_core.F90:1737-1787 iterated
     loop with a shrinking >=nord-deep ghost region). Retest W5/W6.
     **DONE 2026-07-12 — port landed; waves NOT rescued.**  Faithful
     shrinking-ghost evolution for nord in 1..3 (pad divg_d ONCE to a
     nord-deep ghost, each Laplacian shrinks one ring/side — no
     re-copy; nord=1 bit-identical; Fortran-verbatim NumPy reference +
     ghost-ring dependence-cone tests).  Both initial-ghost variants
     selectable end-to-end (`CDGridShallowWaterConfig.
     d_sw5_cross_face_halo`; default zero ghost unchanged).  dd8
     promoted to f64 for nord=3 (del-8 coeff ~1e45 overflows f32).
     **Retest (48 runs, C36 dt=300 x64, matrix IC + blowup gate
     max|u_d|>1000, damp_v=0.02 pinned; nord {2,3} x d4_bg {0.12,0.16,
     0.20} x ghost {zero,cross} x dddmp {0.2,0.5}): NO combination
     stabilizes W5+W6.**  W6 fails 24/24 (blowup day 7.0-10.3 vs 9.0
     at nord=1); W5 fails 22/24 — only nord=3 zero-ghost d4_bg=0.20
     survives 15 d (both dddmp; mass 1.1e-15/3.6e-16) and
     NON-PHYSICALLY (peak max|u| 242-251 m/s vs production 68.6), and
     its W6 twin still blows up day 9.17.  Blowup day is nearly
     INSENSITIVE to nord/d4_bg/dddmp on the zero-ghost family, and the
     Fortran-faithful cross-face ghost makes every case WORSE (W6
     7.0-7.7; earlier with MORE damping) — consistent with the
     Phase-0 modon result that the attenuated ghost pumps the seam
     mode.  Conclusion: the W5/W6 failure is the SEAM-TRUNCATION
     forcing (item 2), not a missing scale-selective divergence sink;
     del-6/del-8 damping cannot rescue the wave class at C36.  This
     result ADVANCES THE FALL-BACK CASE.  Defaults unchanged (nord=1
     zero-ghost preset): the 120-day modon and W2 numbers carry over
     bit-identically (locked by tests), so no re-run was required.
     Full sweep table in the M1b item-1 PR description / session log.
  2. Seam-row truncation forcing (the W2 3.3 m/s/day drift root):
     single-step A+B imbalance harness → term-by-term seam-row
     truncation audit of c_sw+d_sw corner metrics vs the oracle.
     **ROOT-CAUSED AND FIXED 2026-07-12 (commit 8f7ce1f1d) — W5+W6
     RESCUED.**  Method: term-by-term A+B decomposition with exact
     great-circle analytic targets (scripts/tmp/_fb_seam_term_audit.py;
     gnomonic grid lines are great circles → chord-projection tangents
     are exact).  Findings, in order:
     - The exact-input staggering residual R is ~3e-4 m/s/step and
       smooth ⇒ the D-grid staggering itself is innocent; the seam
       imbalance is operator/metric error.  Dominant terms: the
       vorticity-flux delta (T_fx/T_fy) with a 40x seam jump in the
       xfx/yfx area fluxes, plus the a2b PGF delta (T_gz, rows 0-1).
     - Oracle structure check REVERSED the "missing edge overrides"
       hypothesis: fv_arrays.F90:1512 sets ``bounded_domain = regional
       .or. nested .or. duogrid`` — with duogrid, NONE of the Fortran
       edge special-casing fires (d_sw1 ut/vt overrides, d_sw4 vertex
       KE fix, d2a2c edge branch are all ``.not.bounded_domain``- or
       ``.not.dg``-guarded).  The FB chain was already structurally
       oracle-faithful; a probe port of the d_sw1 overrides made the
       imbalance 20x WORSE (uc at duogrid seams is not the ut·sin
       convention those formulas assume).
     - TRUE ROOT: halo METRIC quality.  Fortran-duogrid computes halo
       metrics exactly on the extended grid (fv_grid_tools.F90 duogrid
       branches: ``grid = dg%b_pt`` incl. halos, no
       ``mpp_update_domains``); our ``pad_halo(duogrid=…)`` instead
       Lagrange-remaps the NEIGHBOR face's basis-dependent sin_sg —
       up to 1.8e-2 metric error at the ring-1 seam entries (the values
       every upwind sin selection reads), making the panel-face flux
       metric basis-DISCONTINUOUS (flow-direction dependent).
     - FIX (no new geometry needed): on the extended grid the seam-edge
       metric is continuous, so the exact ring-1 value equals the
       adjacent interior cell's mirror sin_sg channel (identical
       supergrid neighborhood — exact by construction, also in
       Fortran's cos_angle formula).  New shared
       ``fv_tp_2d.pad_sin_sg_upwind()`` applies the four ring
       overwrites (duogrid only; non-duogrid bit-identical, locked by
       tests/core/test_pad_sin_sg_upwind.py); consumers: ``_c_sw``
       mass-flux scaling, ``compute_transport_quantities`` (xfx/yfx),
       ``_deln_flux``.
     - Validation (W2 C36 dt=300 unless noted): single-step seam
       imbalance imb_u 2.81e-2→1.53e-2 m/s/step, xfx seam error
       2.35e7→1.19e6 (40x→1.6x over background); 2-day drift
       3.37→2.96 m/s/day; matrix W2 5d L2 1.54e-2→1.50e-2,
       v_ll_Linf 43.8→26.2.  **Matrix W5 15d and W6 14d now PASS with
       the unchanged M1 preset (mass 4.9e-16 / 9.6e-16) — previously
       blowup day 12.5 / NaN day 9.0, unrescuable by the item-1 48-run
       damping sweep.**  120d C48 modon guard clean (mass 1.2e-15,
       max|u| decaying 20→10 m/s); cosine-bell L2 unchanged
       (1.89e-01).  HONESTY CAVEAT (scripts/tmp/_fb_w5_windseries.py):
       W5 survives the 15d gate but its winds still GROW from ~day 10
       (max|v_d| 30→318 m/s by day 15, max|u_d| 164 vs production
       peak 68.6) — the seam pumping is strongly reduced/delayed
       (physical wave train through ~day 9), NOT eliminated; the same
       first-order smooth vorticity-flux imbalance below is the
       plausible residual driver.  W5/W6 remain calibration-FAIL vs
       production quality; the M1b "waves survive" criterion is met
       only in the matrix-gate sense.
     - REMAINING (follow-ups, not seam-localized): (a) a SMOOTH
       background imbalance (~5e-3–1.2e-2 m/s/step, decaying from
       seam-adjacent latitudes inward) keeps W2 at ~26 m/s v_ll_Linf
       vs production 0.54 — this is the pre-existing "~1.3 m/s/day
       linear drift" follow-up, now the sole W2-parity blocker.
       C36→C72 audit: the vorticity-flux error d_fx converges 2nd
       order (÷3.9) but the induced momentum imbalance T_fx=d_fx·rdy
       only 1st order (÷1.9) — a first-order smooth vorticity-flux
       momentum imbalance of the covariant chain, NOT seam-metric
       infidelity (the exact-target identity residual R converges
       clean 2nd order, ÷3.9);
       (b) rdxa/rdya ring-1 halos are still field-remapped (Courant
       upwind at seam faces; 2nd-order effect, candidate next
       increment); (c) T_gz a2b seam truncation rows 0-1 (~2.6e-3,
       oracle-shared plain-stencil truncation).
  2b. **W2 residual CLASSIFIED + parity reference ESTABLISHED
     (2026-07-12, analysis-only; probes
     scripts/tmp/_fb_w2_dt_convergence.py +
     scripts/tmp/_fb_seam_term_audit.py at C36/C48/C72).**
     Answers to the four M1b follow-up questions, in order:
     - **Q1 — published-Fortran reference: FB is NOT at the oracle's
       own level.**  Mouallem et al. 2023 (JAMES,
       10.1029/2023MS003712 — FV3's own duo-grid paper; W2 α=0, C48,
       dt=512 s, 5 d, MONO; norms = Williamson h-field l1/l2/l∞, same
       definition as our matrix L2) reports: duo-grid l2 ≈ 6e-5,
       l∞ ≈ 1e-4, v-error maps ≈ ±0.05 m/s (Figs 3/5/7); kinked grid
       l2 ≈ 2e-4, l∞ ≈ 3.4e-4, v ≈ ±0.2 m/s with the SAME spatial
       pattern we see (seam-adjacent latitudes decaying inward).
       Convergence rates (Fig 7 legends): duo 2.01–2.03 in ALL norms
       C48→C768; kinked degraded (l2 1.28, l∞ 0.23) but still
       convergent.  Comparison table (W2 5 d, h-l2 normalized /
       v-Linf m/s):
         FB C36 dt300 ........ 1.50e-2 / 26.2   (matrix, reproduced)
         production C36 dt300  4.68e-4 / 0.541  (matrix baseline)
         FV3 kinked C48 ...... ≈2e-4  / ≈0.2    (Mouallem Figs 3,5)
         FV3 duo-grid C48 .... ≈6e-5  / ≈0.05   (Mouallem Figs 3,5,7)
       2nd-order C48→C36 scale factor is only (48/36)² = 1.78:
       production is within ~1.3x of the published kinked level
       (NOT anomalously good — it IS the oracle-kinked level), while
       FB (a duogrid chain, so the duo column is its reference) is
       ~2 orders above.  The M2 parity gate is CORRECTLY referenced.
     - **Q2 — convergence classification: ANTI-convergent.**  Matrix
       protocol at dt=300: C36 1.50e-2/26.2 PASS; C48 2.43e-2/300
       (four equatorial-seam eruptions from ~day 4.5, snapshots_v);
       C72 NaN.  Fixed-Courant probe: C48 dt=225 3.36e-2/286 (same
       eruption — NOT a CFL artifact; Mouallem runs C48 at dt=512);
       C72 dt=150 NaN at ~day 3.5.  dt-independence at C36 (5 d,
       dt 600/300/150): h_L2 1.70/1.50/1.39e-2 — a dt-independent
       SPATIAL floor ~1.4e-2; v_ll_Linf 90.2/26.2/9.32 — the peak v
       is a pumped seam-mode amplitude (per-step del-6 damping means
       smaller dt = more damping per unit time), not a spatial level.
       So the 26 m/s is NEITHER a resolution level nor dt-robust:
       the residual seam forcing grows with n and overwhelms the
       resolution-invariant nondimensional damping preset at n>=48.
     - **Q3 — the "1st-order T_fx puzzle" DISSOLVES: trivial rdy
       scaling, oracle-identical structure.**  sw_core.F90:1935-1944
       updates length-integrated (circulation) increments
       ``u = vt + δ_i(ke) + fy`` with ``vt = u_old·dx`` (1584/1589),
       recovered by ``·rdx/rdy`` downstream (dyn_core.F90:2466/2473)
       — identical to T_fx = d_fx·rdy.  rdy ∝ n, so ANY flux error of
       absolute order p mechanically yields an order p−1 tendency, in
       Fortran too.  3-point audit (C36/C48/C72, single step, dt=300):
       d_fx ÷3.94 (2nd-order absolute = 1st-order RELATIVE to the
       flux magnitude ∝ dy; a fixed-dt semi-Lagrangian time-offset
       would give ÷2, so this is genuine spatial flux truncation),
       T_fx seam ÷1.96 / interior ÷1.94, R ÷3.8.  Budget closure:
       interior T_fx 1.09e-2 m/s/step × 288 steps/day = 3.1 m/s/day ≈
       the observed 2.96 m/s/day drift.  No extra first-order term;
       what remains real is the 1st-order-RELATIVE vorticity-flux
       error itself, which the oracle duo-grid demonstrably does not
       suffer at solution level (its l∞ converges at 2.03).
     - **Q4 — VERDICT: (b) genuinely sub-oracle, cause named.**  A
       smooth first-order (relative) vorticity-flux momentum
       imbalance of the covariant chain (T_fx = d_fx·rdy,
       interior-dominant, seam-adjacent maximum — the oracle-kinked
       error pattern at ~100x amplitude), whose seam-adjacent
       component pumps the equatorial seam mode to eruption at n>=48.
       **M2 flip gate recommendation: DO NOT FLIP (unchanged), and do
       not re-reference the gate to FB's level.**  The W2 target for
       the FB chain is Mouallem's duo-grid column (h-l2 ≈ 1e-4-class
       at C36–C48, v-err ≈ 0.05–0.1 m/s, 2nd-order convergent,
       stable through C768); until the vorticity-flux (fv_tp_2d
       covariant-chain) accuracy is fixed — candidate next increments:
       REMAINING (b) rdxa/rdya ring-1 halos, then a d_fx-targeted
       operator audit vs exact great-circle fluxes at C48+ — FB fails
       W2 on accuracy AND on stability at production resolutions.
       This result ADVANCES THE FALL-BACK CASE.  (Protocol note: runs
       used the worktree tree incl. the concurrent M1b item-3 halo
       vectorization; the C36 matrix lane reproduced the established
       baseline 1.50e-2/26.2 exactly, so the sweep is comparable.)
  2c. **Increment 2b run-down: init-transient vs steady-forcing split +
     the W2 seam mode is a STRUCTURAL INSTABILITY at every resolution
     (2026-07-12, analysis-only; probes scripts/tmp/_fb_w2_lead12.py
     [reads the finished matrix snapshots] +
     scripts/tmp/_fb_w2_seammode_dt.py + _fb_w2_damp_ablation.py; matrix
     protocol held fixed, C36 probe reproduces the 1.50e-2/26.2 baseline
     exactly).**
     - **Lead 1 (init imbalance) REJECTED as dominant.**  Putman-protocol
       (day-1-referenced) h L2 at C36 day 5 = 1.262e-2 vs 1.503e-2
       analytic-referenced — the IC transient is only ~16% of the error;
       the day-1-referenced number is still ~30x above the published
       C45 level.  Transient v peaks ~1.8 m/s in the first 12 h (the
       historical "16 m/s within 6 h" does not reproduce post-seam-fix).
       Covariant-balanced IC (analytic v_cov via ŷ = cosa_u·x̂ +
       sina_u·rot90(x̂), inverted through fb_v_d_to_orthogonal) is
       INDISTINGUISHABLE from the standard IC (max|v| 3.049 vs 3.053
       m/s at day 1; h L2 identical to 4 digits) — no balanced-IC helper
       is warranted; none was implemented.
     - **Lead 2 (diagnostic artifact) REJECTED.**  Native day-5 error
       (31.3 m/s) EXCEEDS the regridded 26.2 — the lat-lon regrid
       smooths, it does not inflate.  Pattern: seam-concentrated,
       max at |lat|<20 (equatorial face seams), decaying to 0.08 m/s
       at the poles; interior (>2 cells from a face edge) only 6.2 m/s.
     - **The v error is an EXPONENTIAL seam mode, not accumulated
       forcing.**  12-h series (v_seam / v_interior split): interior
       saturates ~6 m/s by day 3.5 at C36 while seam rows grow
       exponentially — e-fold rate 1.63/day (C36), 1.78 (C48), 2.09
       (C72).  Extended run: **C36 NaNs at day 7.5**
       (31→60→124→279→813 m/s per half-day) — this corrects the item-2b
       "eruption at n>=48" scoping: the mode is unstable at EVERY
       tested resolution, the 5-day matrix window just ends before
       C36's blowup, and the 26.2 headline is a mid-growth snapshot.
     - **dt-INDEPENDENT (structural, not CFL/splitting).**  C48 dt=150
       late-window e-fold 1.74/day vs 1.78 at dt=300 (dt=75 consistent
       in its window) — growth per unit TIME is fixed by the spatial
       discretization; smaller dt only delays onset via more del-6
       damping per unit time (the item-2b Q2 amplitude observation).
     - **Damping-knob ablation EXONERATES the dampers as drivers and as
       cures** (C48 dt=300 5 d): damp_v=0 byte-near-identical (415.9 vs
       413.6 m/s day 5); d4_bg=0+dddmp=0 still erupts (285 m/s, same
       e-fold; the del-4 div damping actually accelerates the seam mode
       slightly early on — 11.8 vs 9.3 at day 3 — while halving the
       interior error); all-off same.  The instability lives in the
       UNDAMPED core chain (c_sw → p_grad_c → d_sw1-4 + phase-4 PGF +
       covariant entry/exit) on the balanced zonal flow.
     - **Classification (the increment-2b deliverable):** W2 error =
       ~16% init transient (ignorable) + a smooth 1st-order forcing
       that sets the h-L2 floor (linear ~2.7e-3/day at C36, ratios
       1.28/1.43 across C36/C48/C72 ≈ 1st order — the item-2b T_fx
       mechanism) + a seam-localized exponential instability of the
       core FB chain (rate ~1.6-2.1/day, rising with n, dt- and
       damping-independent) that dominates v_ll_Linf from ~day 3 and
       blows up every resolution.  **No guess-fix attempted.**  Next
       increment must root-cause the seam FEEDBACK (not the pump
       amplitude): prime candidates remain REMAINING (b) rdxa/rdya
       ring-1 field-remapped halos feeding Courant/upwind selection at
       seam faces, and the seam-row metric quality of the c_sw/d_sw
       stencils beyond sin_sg (item-2 fixed only the sin_sg channel).
       Confirms + strengthens the item-2b DO-NOT-FLIP recommendation
       and the FALL-BACK case: FB fails W2 on stability at C36, not
       just accuracy at C48+.
  2c-1. **Metric-halo family sweep — DONE 2026-07-13 (commit d00a1f1b4).**
     Extended the exact-extended-grid metric convention (item-2 sin_sg
     pattern) to the whole ring-1 remapped family: rdxa/rdya (Lagrange
     remap was 38% WRONG — a scalar remap cannot know the dxa<->dya swap
     under face rotation), dxc/dyc edge copies, area_c seam/vertex
     scaling, divergence-corner sin/cos.  `create_cubed_sphere_cdgrid`
     runs the metric constructors on the (n+2) extended supergrid
     (`_duogrid_seam_metrics` + `DuoSeamMetricPads`); `_corner_vorticity`
     exact-ring REVERTED (destabilizes W6 — kept as documented edge-copy).
     W2 5d / W5 15d / W6 14d / 120d modon ALL still pass; regression golds
     rebaselined.  **The W2 exponential seam mode is UNCHANGED (C36 NaN
     day ~7) → metric halos are EXONERATED as the seam-instability
     feedback.**  Faithful improvement, KEPT.
  2c-2. **d_fx / vort_hord audit — DIAGNOSED then DISCARDED as un-faithful
     (2026-07-13).**  The W2 *smooth* h-L2 floor (item-2b T_fx mechanism)
     is the absolute-vorticity PPM collapsing to 1st-order upwind wherever
     `zeta+f<=0` — all of the SH for W2 (`zeta_abs = 2 sin(lat)(Omega +
     u0/a)`) — because hord=12 routes through the positive-definite
     `pert_ppm(iv=0)` (tp_core.F90:580).  **BUT the oracle's own default
     `hord_vt=9` (fv_arrays.F90:339) is ALSO positive-definite
     (tp_core.F90:610 -> pert_ppm iv=0): real FV3 has the identical SH
     collapse on W2 and still reaches h-l2 6e-5 (Mouallem).**  A
     sign-agnostic limiter (hord 8/10/11) is therefore a DEVIATION from
     the oracle, not a faithful fix.  Empirical probe
     (scripts/tmp/_fb_vort_hord_w2.py, W2 C36 dt=300):

       | day | hord=12 (faithful) h_l2 | hord=10/8 h_l2 |
       |-----|-------------------------|----------------|
       | 5.0 | 1.52e-2                 | 9.9e-4         |
       | 6.0 | 2.58e-2                 | 2.25e-3        |
       | 7.0 | 7.35e-2 -> NaN d7.5     | 7.9e-3         |
       | 8.0 | —                       | 3.98e-2 (erupting, maxv 677) |

     Sign-agnostic removes the smooth floor (15x better day-5 h_l2) but
     the **exponential seam mode is untouched — same eruption, delayed
     ~1 day, blows up day 6-8 regardless.**  vort_hord edit REVERTED
     (uncommitted, never shipped); finding preserved here + in the probe.

  **=== M1b FINAL VERDICT (2026-07-13): FALL BACK ===**  All three items
  landed; W2 parity NOT reached.  FB is NOT flip-viable: (1) W2 exponential
  seam instability erupts day 5-7 -> NaN at C36, immune to every M1b fix
  (metric halos exonerated 2c-1, vort_hord only delays 1 day 2c-2); (2)
  FB is 2.6-4.3x slower/step; (3) sub-oracle W2 accuracy floor (2 orders
  below Mouallem duo-grid).  The user's pre-authorized fallback trigger
  (2026-07-12) is MET.  Item-1 nord>=2 port + item-2/2c-1 seam-metric
  fixes are FAITHFUL and STAY (they rescued W5/W6/modon and fixed a real
  38%-wrong rdxa remap); the FB core remains the algorithmically-faithful
  reference kept opt-in (`--sw-core fb`), but **production stays the
  DEFAULT cube SW path.  Phases 2-3 (PE/NH) build on the production RK3
  core, not FB.**  M2-M5 below are RE-SCOPED accordingly (pending user
  go-ahead on the pivot).
  3. Step cost: d_sw3 PPM (1.67 ms) + c_sw (1.41 ms) optimization
     (sweep fusion, pad elimination); target <=2x production.  PARKED
     (fall-back: FB is opt-in, not the default → not on the perf-critical
     path; 1 win landed b1c24a86e vectorized halo kernels).
### Phase-1 RE-SCOPED to production-core consolidation (USER DECISION 2026-07-13)

FB failed the flip (M1b FINAL VERDICT). The "single FV3 implementation"
goal now centers on the PRODUCTION RK3 cube SW core (`fv3_sw_tendencies`
in `FV3EdgeShallowWaterModel`), which passes the full Williamson matrix
and is 2.6-4.3x faster. FB is retained as the algorithmically-faithful
opt-in reference (`--sw-core fb`), NOT deleted. Milestones below are
PR-sized units, each with its own codex review.

- [x] **M2 — driver runs the VALIDATED production core. DONE 2026-07-13,
  codex-CLEAN (3-round iterate: NEEDS-FIX → NEEDS-FIX → SOUND).** Commits:
  `2e07ab0f4` (impl) + `5e3bd760b` (round-1: loud reject of silent
  diffusion-knob no-op, grid.n calibration, loud SW guard) + `58d320ecb`
  (round-2: SW rejection factored to `_reject_shallow_water_unrunnable()`
  at both public `setup()`/`run()` entries + `_init_state` backstop, so
  `legoesm run <sw>.yaml` fails cleanly BEFORE the factory scale-guard).
  46 factory + 41 manifest/golden tests pass. Follow-up (codex #4, out of
  scope): `atmosphere.dynamics.create_model` alias + `legoesm test
  williamson` CLI still build CDGrid directly (separate live SW path,
  bypasses the factory) — migrate or label legacy in a later PR.
  The `cdgrid_shallow_water` factory
  branch now builds `FV3EdgeShallowWaterModel` + the resolution-robust
  validated preset `williamson_cli_calibration(gc.resolution)` (matched
  hyperdiff + div_damp + damp_v, stable C24-C48), overriding only the
  driver-exposed knobs (conservation_fixer, fix_mass, time_integrator).
  Tests updated (isinstance Edge + div_damp/damp_v>0); 82 pass
  (test_component_factory 42 + deprecation/grid_factory 40). CDGrid class
  RETAINED (ocean `fv3sw` + deprecated aliases still resolve to it); the
  `atmosphere.dynamics.create_model` alias path left as-is (separate legacy
  convenience dispatcher — follow-up if full consistency wanted). Honest
  caveats: (a) the driver `hyperdiff_scale`/`A_h` knobs no longer reach
  cube SW (validated preset used instead — acceptable, SW is a test
  dycore); (b) the coupled-driver SW *run* path was ALREADY non-functional
  pre-M2 (`_init_state` builds a PE held_suarez state, not SW) — M2
  advertises the faithful core but does not add a SW IC path (out of
  scope); (c) restart migration N/A (no live SW driver checkpoints).
  ORIGINAL INVESTIGATION 2026-07-13 (cavecrew map): the driver
  (`component_factory.py:45-53,300-314`) selects `CDGridShallowWaterModel`
  for ALL cube SW selectors, whose `.step()` calls
  `cdgrid_shallow_water_tendencies` (mass = `dgrid_to_cgrid` +
  (`component_factory.py:45-53,300-314`) selects `CDGridShallowWaterModel`
  for ALL cube SW selectors, whose `.step()` calls
  `cdgrid_shallow_water_tendencies` (mass = `dgrid_to_cgrid` +
  `cgrid_mass_flux_divergence`; momentum = `cdgrid_momentum_tendencies`).
  **But the VALIDATED matrix `production` lane (W2 L2=4.68e-4) uses
  `FV3EdgeShallowWaterModel.step` → `fv3_sw_tendencies` (A-L RK3,
  operators_cdgrid.py:1580) — a DIFFERENT tendency scheme.** So the driver
  currently runs a scheme the matrix never validates. `fv3_sw_tendencies`
  takes the same primitive `(h,u_d,v_d,h_s,cdgrid)` state, so the clean
  route is: point `CDGridShallowWaterModel.step` at `fv3_sw_tendencies`
  (keeps the driver + coupler + ocean-barotropic state type
  `CDGridShallowWaterState`). CHANGES driver + ocean-barotropic ("fv3sw")
  numerics → needs full atm-cube-SW matrix (via driver) + ocean
  barotropic 9/9 revalidation + possible div_damp/damp_v retune; codex
  (numerics). NOT a trivial wiring swap.

  **M2 CORE-EQUIVALENCE VERDICT (2026-07-13, codex-reviewed BEFORE deciding
  per the standing rule): the two cores are NON-ISOMORPHIC different
  discretizations, and Edge is the justified canonical core.**
  - **Different staggerings (codex CRITICAL, confirmed):** CDGrid winds are
    corner-corner `(6,n+1,n+1)` (2·(n+1)² per face); Edge is edge-midpoint
    split-D u_d `(6,n,n+1)` + v_d `(6,n+1,n)` (2·n(n+1) per face).
    `dgrid_to_cgrid_core` (operators_cdgrid.py:422) expects corners; running
    CDGrid on the Edge IC crashes (shape `(6,36,36)` vs `(6,37,36)`). So
    "point CDGrid.step at fv3_sw_tendencies" is INVALID — not a shared grid.
  - **W2 is NOT a differentiator (codex caught my confound):** with
    div_damp=0 + hyperdiff, CDGrid W2-5d C36 area-L2 = 2.36e-4 (hd 2×) …
    4.38e-4 (hd 8×) — matches/beats Edge's 4.68e-4; cube-edge ratio drops
    2.069 → 0.80 (PASS). The earlier 2.069 "artifact" was under-hyperdiff,
    not staggering. div_damp DESTABILIZES the corner staggering (×8 NaNs).
  - **W5 is the decider — CDGrid cannot pass it (scripts/tmp/_fb_cdgrid_w5_tune.py):**
    W5-15d C36 peak|u| = 554 (hd 2×) → 1254 (hd 32×) m/s; A_h 1e4-5e5 gives
    526-578; div_damp NaNs. NO supported CDGrid knob reaches <100 m/s
    (Edge passes; physical peak ~40-68). CDGrid structurally lacks Edge's
    `damp_v` del6_vt_flux scale-selective VORTICITY sink (needs edge-midpoint
    winds — fv3_del6_vt_flux.py:257), the exact stabilizer the cube W5/W6
    wave class needs. Scalar hyperdiff on winds does not target the enstrophy
    mechanism. **This meets codex's decision criterion: no stable tuned
    CDGrid point exists → Edge is the justified canonical core.**
  - **Scope corrections (codex, reduce M2 cost):** (a) ocean migration is
    LIGHT — the `fv3edge` barotropic adapter already lifts cell-centre→Edge
    winds and runs end-to-end in `OceanModel` (tests/ocean/unit/
    test_fv3edge_barotropic.py); only raw-SW restart checkpoints need shape
    migration (io/state_checkpoint.py:208). (b) The factory swap must also
    wire the VALIDATED config (`div_damp`+`damp_v`; component_factory.py:300
    currently omits both), not just change the class. (c) CDGrid is NOT
    deprecated and is retained (ocean default `fv3sw` still selects it until
    M4b); M2 only re-points the ATMOSPHERE cube-SW driver default to Edge.
- [ ] **M3 — W2 cube v-imprint on the canonical Edge core (RE-SCOPED
  2026-07-13; premise was stale post-pivot).**
  PREP FINDING: the original M3 premise ("fix `d2a2c_vect` orthogonal-
  through-covariant defect") described the OLD production = CDGrid
  (`dgrid_to_cgrid`), now retired from the driver. **The canonical Edge
  `fv3_sw_tendencies` DEFAULT path does NOT use `d2a2c_vect`** — that call
  is only under the `use_split_mass_momentum_integration` /
  `use_fv3_dsw1_mass_transport` opt-ins (default False). The Edge default
  D→C is `fv3_d2cc` (operators_cdgrid.py:1498 — "orthogonal-rotation
  convention", i.e. it already treats the model's orthogonal winds
  correctly) + `fv3_cc2c` (cc→C-face avg with the non-orthogonality
  projection). So the orthogonal-vs-covariant defect likely does NOT apply
  to the Edge default path. BEFORE any fix: (i) diagnose whether the Edge
  core actually carries a cube W2 v-imprint worth fixing (visual W2 v-wind
  + `visual_regression --check` on the Edge lane; matrix Edge W2
  v_ll_Linf=0.541), (ii) if so, locate its SOURCE in the Edge path
  (`fv3_d2cc`/`fv3_cc2c`/momentum A-L gradient / vorticity), NOT assume
  `d2a2c_vect`. M3 may shrink to "no actionable Edge v-imprint defect"
  or re-target a different operator; codex review on whatever fix lands.

  **M3 DIAGNOSIS DONE 2026-07-13 → NO ACTIONABLE DEFECT (no code change).**
  Probe scripts/tmp/_fb_edge_w2_imprint.py: canonical Edge W2-5d C36
  (FV3EdgeShallowWaterModel + williamson_cli_calibration), same cube-edge
  artifact ratio the CDGrid harness uses
  (williamson_diagnostic.cube_edge_artifact_ratio, PASS < 2.0):
  - **v_north (W2 α=0 ⇒ analytic v=0 ⇒ the v field IS the error): ratio
    0.780, INTERIOR-dominated (interior max 1.055 vs boundary 0.647
    m/s).** No edge-localized v-imprint on the Edge core — the ~0.54
    v_ll_Linf is a SMOOTH sub-oracle residual (Mouallem-kinked-level),
    not a convention-fixable cube imprint.
  - h-error: ratio 2.316 (marginally > 2.0) but on a tiny error (W2 L2
    ~2.4e-4); the known smooth sub-oracle residual, not worth a numerics
    change to the canonical core.
  The v-imprint the original premise worried about was the RETIRED CDGrid
  path's (2.069 h-artifact on its corner-corner staggering). Edge's
  `fv3_d2cc` (orthogonal-rotation) + `fv3_cc2c` (non-orth projection)
  already treat the orthogonal model winds correctly. **M3 CLOSED: the
  consolidation's "one real accuracy win" (fix d2a2c_vect) does NOT apply
  to the canonical Edge core — it was CDGrid-specific.** No fix ships.
- [x] **M4 — flag collapse (dead-opt-in deletion, NO model merge). DONE
  (9cdda8c0c), codex-clean.** Deleted 10 provably-dead default-False flags
  from `CDGridShallowWaterConfig` + every gated branch/param + the
  now-unreachable `fv3_csw_tendencies` + orphaned imports (−1560 net LOC).
  Bit-identity proven byte-for-byte: W2 L2=4.68e-04/Linf=4.44e-03/
  v_ll_Linf=5.41e-01, W5 mass=6.47e-16, W6 mass=1.53e-15 all EXACT vs
  baseline; FB lane 6/6. Live-path residual flag refs = 0 (the only hits
  are frozen `scripts/tmp/` archive probes, gitignored, never in CI, + one
  path-string in an import manifest). Kept `CDGridShallowWaterModel` +
  FB + `FV3EdgeShallowWaterModel` as distinct classes (no clean merge —
  states are non-isomorphic). Codex re-review run **inside the worktree**
  (the first run mis-read the primary tree on a different branch → stale
  M2-era findings; disregarded).
- [x] **M4b — ocean barotropic: KEEP BOTH, retire nothing. DONE (no code),
  codex-backed (NEEDS-MORE-EVIDENCE for migration).** Decision: do NOT
  migrate ocean `fv3sw` (CDGrid corner) → `fv3edge` (Edge). Rationale:
  (1) ocean already has BOTH adapters wired — `fv3edge` runs end-to-end
  through `OceanModel` (`test_fv3edge_barotropic.py:70`, C24/0.5d
  finite=True, nonzonal_fraction=0.019); (2) `fv3sw` is a *selectable
  default*, not a coupling necessity; (3) migration would force a
  restart-schema migration (`state_checkpoint.py:208` validates field
  shapes) for **no proven ocean benefit** — no controlled evidence
  `fv3edge` beats `fv3sw` in the ocean barotropic regime (external gravity
  waves + geostrophic adjustment, not the Rossby-Haurwitz W5 class that
  drove M2). Ocean legitimately offers a_grid/c_grid/fv3sw/fv3edge; keeping
  the working default is the lazy + evidence-based call. M4 kept
  `CDGridShallowWaterModel`, so no class deletion is blocked.
- [~] **M5 — ship, SPLIT into two PRs (2026-07-13).** M4 (dead-flag
  deletion) does NOT cleanly cherry-pick onto current main: it was authored
  against the M1b d_sw5 state (nord 1..3 shrinking-ghost + `d_sw5_cross_face_halo`
  config field), but main carries the pre-M1b `d_sw5_corner_divergence`
  (nord>1 RAISES; no such config field), and the M1b commits are NOT being
  landed. M4's byte-for-byte bit-identity proof was measured vs the branch
  baseline 806d74bd2 (which INCLUDES M1b), so it does not transfer to main —
  re-deriving the deletion on main is a distinct change requiring its own
  fresh W2/W5/W6 bit-identity proof. Therefore:
  - **PR A (this one) — M2 + program doc.** The actual consolidation: the
    driver cube-SW factory now returns the matrix-validated
    `FV3EdgeShallowWaterModel` (was building the un-validated
    `cdgrid_shallow_water_tendencies` corner core). Codex-clean (3-round
    iterate to SOUND on the source branch; the diff cherry-picks onto main
    with zero conflicts, so it is byte-identical to the reviewed change).
    46/46 factory tests + 21/21 SW driver-guard tests green on main.
  - **PR B (follow-up) — M4 dead-flag deletion, re-derived on main.** Delete
    the 10 provably-dead default-False flags + `fv3_csw_tendencies` on main's
    OWN file versions (surgical, decoupled from M1b), with an independent
    bit-identity proof measured against main. Tracked as a separate cleanup.
  Full atm matrix (all grids), ocean matrix, MPI/SPMD parity, and visual
  gates run against PR A before merge.
  **Justification caveat (codex confound caution):** the PR must NOT claim Edge
  is *intrinsically* numerically superior to CDGrid on W2/W5 — that
  comparison is damping-knob-confounded (W2 edge-artifact 2.069→0.796 when
  hyperdiff is matched; CDGrid W5 max_speed observed finite ~358 m/s, not
  the earlier 554-1254). The DEFENSIBLE canonical-Edge justification is
  **provenance**, not intrinsic superiority: Edge is the matrix-validated
  cube-SW core (W2/W5/W6), and CDGrid structurally cannot use the `damp_v`
  del6_vt scale-selective vorticity sink (Edge-shaped winds only). Route
  the driver default to the validated core; retain CDGrid opt-in. Frame M5
  claims exactly that way.

## Phase 2 — hydrostatic PE on the FV3 architecture

- [ ] **M1 — state restructure**: split-edge D-grid staggering
  (`u_d (6,n,n+1,k)`, `v_d (6,n+1,n,k)` replacing the corner-corner
  `FV3HydrostaticState`), halo/vertex ops, restart migration.
- [ ] **M2 — FB stepping**: c_sw → p_grad_c → d_sw per level with the
  Phase-1 SW core reused per layer; Lin PGF live (`_fv3_lin_pgf`), dead
  `d2_bg>0` gate fixed (`nord>0 or d4_bg>0`).
- [ ] **M3 — vertically-Lagrangian coordinate + remap** (fv_mapz port).
- [ ] **M4 — shared mass fluxes for T/moisture/tracers** (replaces the
  first-order post-RK3 tracer split).
- [ ] **M5 — validation**: baroclinic/rotated/HS/DCMIP matrix, AMIP smoke,
  codex, PR.

## Phase 3 — nonhydrostatic

- [ ] delp/delz/zh moving-layer state, Riemann solver + vertical remap
  chain (nh_core.F90/nh_utils.F90 port) replacing the θ′/ρ′ fixed-height
  acoustic dycore; DCMIP NH cases; codex; PR.

## Standing rules

- One milestone = one PR-sized change, codex adversarial review each.
- Truth tiers outrank oracle-matching (Phase-0 precedent: the faithful
  d_sw5 ghost was measurably worse and stays opt-in).
- Never A-grid. Production stays bit-identical until its milestone
  deliberately retires it.
- Oracle: `~/Documents/Code/FV3/atmos_cubed_sphere-symmetryclean`.
- Session memory: `fv3-sw-review-2026-07-10` (Phase 0),
  `fv3-consolidation-program` (this program).
