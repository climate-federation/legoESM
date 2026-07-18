# Phase-4c oracle: authoritative Mouallem/Xi-Chen duo-grid FV3 (Zenodo 8327578)

Phases 4a (c_sw) and 4b (d_sw) certified legoESM's loop-faithful reference
field-by-field against a VERBATIM Fortran extraction from
`GFDL_atmos_cubed_sphere @ 6f658bd0` — the **plain** (non-duo) FV3.  Phase 4c
re-targets the oracle to the **authoritative duo-grid FV3** the campaign
actually implements, and adds the cross-grid grid-imprinting battery.

## The oracle (Zenodo 8327578)

Mouallem 2023 (JAMES), *"Implementation of the novel Duo-Grid in GFDL's
FV3 — Code and simulations files"* (3.6 GB), fetched to
`/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/` (job 9033356).
Contents:

- `atmos_cubed_sphere-symmetryclean.zip` → the **definitive duo-grid FV3
  source** (extracted to `Code/FV3/duogrid_symmetryclean/`).  This is the
  `symmetryclean` tree memory flagged as the authoritative provenance
  (Xi Chen, Princeton — the FV3 duo-grid author).
- Idealized simulation reference outputs, one dir per case, each with
  `rundir/atmos_daily.nc` (reference solution) + `input.nml` (config):
  - `C{48,96,192,384,768}.sw.case2.alpha{0,45}[.duo].hord{5,6,8,10}` —
    **Williamson-2** (solid-body rotation), duo vs non-duo, the primary
    grid-imprinting case.
  - `.sw.case{5,6,8}.*` — further shallow-water cases.
  - `.nh.case-13.*` — 3-D baroclinic wave.

## Re-pin findings (authoritative vs what phases 3-4 pinned)

Established this phase (`reconcile_production_csw.py`, sbatch on `glab`):

1. **Production grid geometry is BIT-EXACT to the certified builder.** The
   JAX `create_cubed_sphere_cdgrid(create_fv3_native_cubed_sphere(...))`
   tile-1 `area` matches `build_fv3_native_gridstruct` to 8.6e-14 (face 3 /
   rot90=1); `sin_sg`/`sina_cell` agree to 2.7e-8 (cdgrid double vs the
   certified longdouble `compute_fv3_native_angles` — sub-1e-7, not
   bit-exact; follow-up: adopt the longdouble angle build in the cdgrid).
   `cosa_cell` differs only by the rot90-induced sign flip of the signed
   non-orthogonality angle (a comparison artifact, not a grid difference).

2. **The duo-grid c_sw/d_sw DIFFER from the plain-FV3 6f658bd0 extraction.**
   The authoritative c_sw adds `flagstruct%duogrid` branches: it calls
   `divergence_corner_duo` (not `divergence_corner`), **skips**
   `fill2/fill_4corners` (the duo halos carry real cross-face data), and
   takes the `bounded_domain .or. grid_type>=3 .or. duogrid` KE/vorticity
   branch — i.e. **no `sin_sg` panel-edge special-casing**, because the
   duo-grid supplies genuine cross-face edge winds.  legoESM's PRODUCTION
   `fv3_sw_core._divergence_corner_duo` already ports this
   (sw_core.F90:2345-2447), so the production solver is duo-grid; the
   phase-4a/4b certified *reference* implements the plain branch.  Full
   duo-grid fidelity requires certifying the duo branches.

3. **The authoritative `fv_duogrid.F90` / `global_grid_gen_k2e.F90` differ
   from the luanfs mirror phase-3 pinned against** (fv_duogrid 212 lines —
   largely the mirror inlining geometry helpers the authoritative imports;
   `global_grid_gen_k2e.F90` (775 ln, Xi Chen) is a purpose-built k2e
   generator vs the mirror's monolithic `global_grid.F90` (2461 ln), same
   six-stagger structure `get_loc_{x,y}[_b,_c_x,_c_y,_d_x,_d_y]`).  Same
   author, same method — phase-3's k2e tables are expected numerically
   faithful, to be re-verified against the definitive generator.

## Remaining P4c work (loop, each gated by codex review)

- **Re-verify phase-3 k2e** against `global_grid_gen_k2e.F90` (compile,
  dump tables, compare to `compute_fv3_native_k2e`).
- **Certify the duo branches** of c_sw/d_sw (`divergence_corner_duo`, the
  no-corner-fill + no-edge-special-case paths) — the branch the production
  solver actually runs — as a duo-grid one-step oracle.
- **Grid-imprinting battery**: run legoESM's FV3-native duo-grid SW dycore
  on Williamson-2 (C48, alpha0/45, hord6/8) and compare to Mouallem's
  `atmos_daily.nc`; check the cube imprint is absent (the whole point of
  the duo grid) and that it stays close to lat-lon / MPAS.  Then W5/modon
  and Held-Suarez.

## P4c battery results — C36 Williamson-2 imprint ladder (2026-07-15)

Run by `scripts/cluster/fv3_native/sw_imprint_battery.sbatch` (job
9035783).  Williamson-2 has analytic `v == 0` everywhere, so the
lat-lon-regridded `v_ll_Linf` is a pure grid-imprint metric.  Wiring
landed via `--fv3-native-grid` / `--fv3-native-angles` (matrix runner) +
the FB-lane ED build (`_fb_cube_sw_model`) + `FV3FBShallowWaterModel(...,
fv3_native_angles=)`.

| row | solver | grid            | v_ll_Linf | L2       |
|-----|--------|-----------------|-----------|----------|
| A   | A-L    | equiangular,noduo | **0.54**  | 4.68e-04 |
| B   | A-L    | ED + duo        | 22.5      | 4.25e-02 |
| C   | FB     | equiangular,duo | 43.8      | 1.54e-02 |
| D   | FB     | ED + duo        | 58.9      | 1.48e-02 |
| E   | FB     | ED + duo + native angles | **NaN (blew up)** | nan |
| —   | ref    | lat-lon         | —         | 1.41e-03 |
| —   | ref    | MPAS (ico5)     | —         | 1.25e-04 |

Closest-to-isolated rungs (still whole-config, NOT single-metric — see
finding 2): C→D = the ED-native grid config (43.8→58.9, WORSE); D→E =
native seam angles (→ blow-up).

**Findings (largely negative for the native-grid FB path as assembled):**

1. legoESM's **tuned production A-L solver on the equiangular grid (row A)
   is already the best cube and is close to lat-lon / MPAS** (L2 4.68e-4 vs
   lat-lon 1.41e-3, MPAS 1.25e-4).  The science goal — cube W2 close to
   other grids — is met on the *production* configuration.
2. The FV3-native **ED grid config does NOT reduce imprint**: C→D (FB core,
   equiangular-duo → ED-native-duo, same solver + M1 config) goes
   43.8→58.9 — WORSE.  CAVEAT: no rung is a single-*metric* isolation.
   C→D switches the whole ED-native grid config — the ED gnomonic metrics
   AND their *mandatory* `k2e_nord=4` duo tables (order-2 is the wrong
   interpolant on ED, ~4e-2 coeff error, so ED forces order 4;
   `cubed_sphere.py:462`), which are inseparable from the metric family.
   A→B (A-L 0.54→22.5) is bundled differently (it also flips no-duo→duo).
   So the rungs compare whole grid/solver *configs*, giving directional
   evidence (every ED/FB config here is worse than production-A-L-
   equiangular), not clean per-knob attribution.  The equiangular-tuned
   A-L config in particular does not transfer to ED (`cubed_sphere_cdgrid
   .py:639`).
3. Row A (0.54) vs row C (43.8) is a **~80× gap between the tuned
   production A-L config and the FB M1 config** — but this is a *bundled*
   difference (solver A-L↔FB, no-duo↔duo halo, AND damping/calibration),
   so the observation is valid but is NOT attributable to the FB core
   alone.  The M1 preset is a coarse, un-tuned damping, not a calibrated
   production config.
4. **Native seam angles + the current FB halo blow the FB core up (row
   E → NaN by step 200 ≈ 0.69 d).**  This is the closest-to-isolated rung:
   D and E differ ONLY in `fv3_native_angles`, same ED grid + FB core + M1
   config.  D (legacy angles) survives the full 5-day window (v_ll_Linf
   58.9); E (native angles) goes NaN by step 200 — ~10× faster than the
   baseline FB equiangular seam mode (which NaNs ~day 7.5 per
   `fv3_single_implementation_program.md`).  So native cross-face seam
   angles *with the FB `d_sw5` zero-ring halo* are inconsistent →
   instability, and they sharply ACCELERATE the known FB seam mode.
   HYPOTHESIS (untested here): FV3 pairs the native seam angles *with* the
   faithful d_sw5 cross-face halo, and legoESM currently runs the stable
   zero-ring approximation instead (the faithful attenuated ghost itself
   destabilises the modon run; `fv3_sw_core.py:3205`) — so the matched
   faithful halo may be the missing piece.  D→E does NOT itself test the
   faithful halo.

## Campaign EXIT battery part 1 — full SW matrix (2026-07-16)

`scripts/cluster/fv3_native/sw_full_battery.sbatch` (job 9043862): every
SW case × every grid, PRODUCTION configuration.  **24/24 PASS.**

| case | cube C36 | latlon | MPAS (ico5) | spectral (T21) |
|------|----------|--------|-------------|-----------------|
| W2 (L2)          | **4.68e-4** | 1.41e-3 | 1.25e-4 | 1.8e-7 |
| W5 (mass drift)  | 9.7e-16 | 3.2e-16 | 0.0     | 3.2e-16 |
| W6 (mass drift)  | 1.3e-15 | 1.9e-16 | 0.0     | 9.6e-16 |
| modons 100d (mass)| 7.3e-16 | 0.0    | 1.8e-16 | 7.3e-16 |
| cosine bell (L2) | 0.20    | 0.13   | 0.62    | 0.38 |

- Cube W2 error is BETTER than lat-lon and the same order as MPAS; the
  W2 v-wind imprint is 0.54 m/s (production A-L calibration).
- Mass is machine-zero on every grid for W5/W6/modons; the colliding
  modons run the full 100 days on the cube cleanly.
- The **no-artifacts gate passed**: `visual_regression.py --check` gives
  SSIM=1.0000 (min 0.985), hamming=0 (max 4), edge_ratio == reference
  exactly (1.349).

The Williamson + colliding-modons + no-artifacts components of the
campaign exit criterion are MET on the production configuration.
Held-Suarez cross-grid (part 2) is running (job 9044126 + supplementary
9044520/9044521).

### Visual edge-artifact inspection (2026-07-16, all SW + completed HS)

Every cross-grid comparison PNG inspected (per the visual-verify rule —
norms alone never certify edges):

- **W2 v-wind** (canonical: exact v==0): cube shows the KNOWN smooth
  4-fold imprint + polar-seam maxima at the calibrated ~0.5 m/s — smooth,
  large-scale, no sawtooth, no face-line discontinuities; matches the
  committed visual-regression reference exactly (SSIM=1.0000, hamming=0).
  MPAS shows its own 12-pentagon dipole signature; latlon/spectral ~0.
- **W5 wind_speed / W6 height**: identical synoptic structure on all four
  grids; cube panels smooth, no face lines (cube slightly more diffuse —
  the calibrated damping).
- **Colliding modons, day 100 vorticity**: NO face-seam eruptions (the
  historical #521 eruption stays fixed through the full collision +
  return).  Cross-grid dissipation spread (ico keeps the tightest
  dipoles, cube dispersed-but-smooth, spectral dissipated) is scheme
  diffusivity, not an artifact.
- **Cosine bells**: the bell crosses four cube faces and returns compact;
  no seam tearing (spectral shows the expected T21 Gibbs ripples).
- **Held-Suarez cube C36 (sigma, 200 d)**: classic HS climate; v-field
  longitudinal structure is TRANSIENT eddies (moves between snapshots, no
  stationary lock).  ONE quantified grid signature: a stationary
  **wave-4 modulation of the lowest-level equatorial easterlies, amplitude
  0.19 m/s** (crests within ~10 deg of the cube corners; wave-8 harmonic
  0.035 m/s; waves 1/3/5 = 0) against ~3.5 m/s background — a ~5% smooth
  corner imprint, no discontinuities.  The high-latitude dotted moire in
  the *native* scatter PNGs is plotting sparsity, not model signal
  (absent in the regridded fields).

Summary: no destructive edge artifacts anywhere; the residual cube
signatures are the two documented smooth imprints (W2 v ~0.5 m/s;
HS equatorial u wave-4 0.19 m/s ~ 5%), both at the calibrated C36 level.

### Campaign EXIT battery part 2 — Held-Suarez cross-grid (2026-07-16)

200-day Held-Suarez, matrix defaults (jobs 9044126 + 9044520/21):

| grid | cases | verdict |
|------|-------|---------|
| cubed_sphere C36 | sigma + hybrid + topo | **3/3 PASS** (mass 1.6e-16 / 1.6e-16 / 1.0e-14) |
| latlon 72x144 | sigma + hybrid | **2/2 PASS** (mass 1.6e-16 / 0.0) |
| spectral T21 | sigma + hybrid + topo | **3/3 PASS** (mass ~5e-16; max\|v\| 40-59 m/s jets) |
| icosahedral ico5 | — | INCOMPLETE (10 h walltime insufficient for the
MPAS hydro path; relaunched unbuffered as job 9067601 to diagnose
JIT-vs-throughput) |

Cube climate verified visually: classic HS structure, transient eddies
(no stationary lock), the one quantified signature being the 0.19 m/s
equatorial wave-4 imprint documented above.  With the SW battery
(24/24), the visual/artifact gate, and HS cube+latlon+spectral all
passing at machine-zero mass drift, the exit criterion is met on the
production configuration for every grid whose runs completed.

### W2 wave-pattern benchmark vs the authoritative FV3 references

The W2 v "wave-like" error field was benchmarked directly against
Mouallem's Zenodo `atmos_daily.nc` references (day 5, same 181x360 grid,
hord6; `fv3_recon/w2_wave/w2_v_vs_fv3_refs.png`):

| day-5 W2 v-error       | max\|v\| (m/s) | rms (m/s) |
|------------------------|---------------|-----------|
| legoESM production C36 | 0.54          | 0.098     |
| FV3 PLAIN C48 hord6    | 1.38          | 0.097     |
| FV3 DUO   C48 hord6    | 0.024         | 0.010     |

- The pattern is the quasi-stationary cube-harmonic error (midlat wave-4
  amplitude 0.10 m/s, phase drift ~5 deg per half-day).  Authentic PLAIN
  FV3 shows the SAME grid-locked wave-4 class at IDENTICAL rms and a
  2.6x LARGER corner peak — despite running at the finer C48 (the
  resolution asymmetry favours FV3, so the conclusion is conservative).
- legoESM's one distinctive component: smooth polar wave-2 arcs
  (0.107 m/s) where FV3-plain is polar-clean but corner-spiky.
- The DUO grid eliminates the whole pattern (10-40x cleaner) — the
  quantified payoff of the native-duo path this campaign certifies
  kernel-by-kernel (remaining assembly blockers documented above).

## Duo D-grid finding (2026-07-16): not single-tile certifiable

Attempting the duo d_sw analog of the (certified) duo c_sw exposed a hard
scope boundary in the authoritative symmetryclean pipeline:

1. **Inter-panel flux averaging mid-sequence.**  The duo dyn_core averages
   the delp/temperature fluxes across panel edges BETWEEN its d_sw1 and
   d_sw2 stages (`mpp_get_boundary` + `0.5*(own+neighbor)`;
   dyn_core.F90:853-900 — "averaging ... is fundamental").  This is a
   6-face communication in the middle of the D-grid step — impossible in
   a single-tile oracle.
2. **Undefined panel-edge workspace reads.**  With the reference-run flags
   (duogrid=T, bounded_domain=F — verified from the Zenodo rundir
   `input.nml`), the `.not.bounded .or. .not.duogrid` guards are
   always-true, so d_sw1's edge blocks execute and read ut/vt panel-edge
   cells the duo interior loop never writes; dyn_core's utt/vtt are
   uninitialised stack arrays (research-grade code — there is a literal
   `!!!!! CODE CRASHES HERE !!!!!` comment at dyn_core.F90:879).  The
   Fortran values there are compiler-dependent.  Empirically, running the
   verbatim duo gates single-tile NaN'd the interior transport through
   exactly that ut(0,*)/vt(*,0) → yfx/ra_y chain.

Consequence: `d_sw(duogrid=True)` in the port RAISES (fail-loud); the
verbatim duo gates remain in the code as the base for **per-stage** duo
certification (d_sw1/d_sw3/d_sw5 driven with fully-specified inputs) and
a legoESM-side inter-panel flux-averaging analog (6-face exchange
infrastructure, like the P4b corner-B-halo precedent).  The duo c_sw cert
is unaffected (its duo branches fully define every read cell).

**Consequence for "full FV3 faithful portability":** the *kernels* are
certified bit-exact (c_sw / d_sw / divergence_corner_duo), and the
*production* cube already meets the science goal on the equiangular grid.
But the *assembled native-grid FB path* (ED + native angles, currently with
the FB zero-ring `d_sw5` halo) is NOT yet consistent — row E blows up.  The
proposed next brick (a hypothesis, not established by this battery) is to
port and stabilise the native-angle ⇄ faithful-d_sw5-halo pair *together*,
rather than swapping the grid under the A-L solver (which is not FV3's
scheme and is tuned for equiangular).

## Duo per-stage certification + integrated stepper (2026-07-17)

**Per-stage TRANSLATION certificates** (bit-exact uint64 vs verbatim
symmetryclean Fortran extracts on the committed C12 inputs; every
oracle input-hash + extract-SHA enforced; all codex-reviewed to SHIP):

| construct | scope | test |
|---|---|---|
| c_sw duo | full duo branch, 5 outputs | test_fv3_native_c_sw_duo |
| d_sw1 duo | transport stage, 16 tokens, defined-workspace shim | test_fv3_native_dsw1_duo |
| d_sw2 duo | delp/pt update, chained | test_fv3_native_dsw2_duo |
| d_sw3 duo | KE fluxes, WMP-delta threaded | test_fv3_native_dsw3_duo |
| d_sw4 duo | 4-corner KE fix | test_fv3_native_dsw4_duo |
| d_sw5 duo | damping/KE/vort transport, 13 tokens, single-face raw-KEE chain | test_fv3_native_dsw5_duo |
| d_sw6 duo | final circulation winds | test_fv3_native_dsw6_duo |

**Integrated six-face duo stepper** (`fv3_native_duo_stepper.py`,
codex SHIP): certified stages + both inter-panel averaging analogs
(dyn_core 853-900 C-ring, 968-1020 BGRID — truth-tiered vs a
geometry-derived 24-edge contact table) + geopk(SW)/p_grad_c/
one_grad_p ports.  Balanced W2 at C12: mass exact, interior departure
0.6 m/s @2 h, edge saturating ~14 m/s (interim index-copy exchanges).

**Duo-target quantification** (job 9075107): C24 5-day W2 stable,
v_ll ~17 m/s steady; gate score 179x/101x the C24-scaled duo envelope
(0.1416/0.0579).  The duo edge-cleanliness target requires the full
extended-halo consistency bundle.

**Ext-bundle measurements** (all flag-gated OFF): extended-lattice
gridstruct (interior bitwise, halos real) + per-stagger k2e remaps
(halo accuracy 0.42 -> 0.0025 on analytic fields) SHIPPED; but every
lightweight in-stepper swap measured WORSE than the coherent interim
(position-only 22/21.6; basis-corrected D 17.4/27.9 vs interim
8.3/14.0) — piecewise approximation of the duo ext machinery injects
errors comparable to those it fixes.  Evidence-driven roadmap: adapt
the stepper (reference numbering) to the certified production
ext machinery (create layout: ext_vector_dgrid basis='covariant',
pad_halo(duogrid)) via the _GNOMONIC_ED_FACE_PERM/_ROT adapter, all
fields + metrics switching together.

**FAITHFUL EXT PORT (4e3fe4a7e + 3474ecf24)**: `fv3_native_ext_vector`
ports the authoritative machinery outright in the stepper's reference
numbering (no create-layout adapter): ext_scalar A/B (own-stagger k2e
rings + 9-slot corner Lagrange) and ext_vector D/C via the upstream
lat-lon intermediary — c2l_ord2(_cgrid) with exact a11..a22 on kinked
mpp-state metrics, geographic exchange on the ng=4 lattice
(set_bd_ext_duo), A-table cube_rmp rings 1..4, corner Lagrange, then
cubed_a2d/a2c projection onto the a2stag ext bases and the S/N-then-W/E
strip writes (fv_duogrid.F90:626-975).  Upstream itself rejects
per-stagger vector remapping as noisier (:678-681) — consistent with
the earlier lightweight-swap measurements.  Halo strips vs
analytic-through-identical-projection truth: 0.152 m/s at C12.
**C12 ablation (SB5a protocol, one FLAG — bundle-level attribution
only: the flag switches halo metrics, A/B scalar exchanges, D/C vector
exchanges, bases and corner handling TOGETHER; it does not isolate the
vector port)**: edge du48 13.99 → 8.25 (−41%), du12 8.28 → 5.01,
interior du12 0.54 → 0.35, ddelp12 1.42% → 0.97%, mass exact; interior
du48 1.33 → 1.86 (small degradation, C24 arbitration pending).  Bugs the analytic gate caught:
create-vs-reference face layout in the ext bases (ext_parity_lonlat_ref
fixes), staggered corner abscissae needing the ng=4 A lattice, and the
c_sw sin_sg(5) tiny-floor patch poisoning the a-matrix (recomputed from
inner(ec1,ec2)).

**EXT-BUNDLE CERTIFIED (codex r4 SHIP, 667ad2b0d)**: independent
Fortran projection oracle (fv3_extproj_*: verbatim a2stag_metrics +
cubed_a2d/a2c; bases 1e-13, projections 1e-12, mutation-discriminated,
sha-enforced fixture) + guards/provenance closures.  Corner-fill wedge
accuracy (codex probe, C12): lagrange default 0.084 m/s; the a2d
variant carries ~3.9 (ring-4 index-copy contamination) and is labeled
a NONFAITHFUL measurement variant.  **C24 W2 gate: ext bundle
23.70/4.08 = 251x/106x of the duo envelope vs interim 179x/101x — the
max sits at CUBE VERTICES (interim: edge-midlat), so the ext swap
cleans edges but excites vertices; wedge VALUES exonerated (0.084),
open suspects: divgd B-corner feed into the nord damping, C-vector
wedges, ext halo metrics under d_sw5's full-domain vorticity prep.**
