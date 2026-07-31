# Cubed-sphere faithfulness audit — 2026-07-31

Full matrix sweep of the cubed-sphere lanes against the FV3 duo-grid Fortran
oracle, run from a **pinned worktree** at `e1e7ef2c5` so that concurrent commits
could not change what later legs imported (a shared checkout had already
corrupted an arm campaign earlier the same day).

Status: **PARTIAL**. The shallow-water lane and the cube-imprint visual gate are
complete; the hydrostatic lane is partially reported; the non-hydrostatic lane
and the oracle twins were still running when this note was written. Numbers
below are only those actually measured — pending lanes are marked as such rather
than predicted.

## Result: the norms pass, and that is not the whole answer

The question this audit was asked to settle was "are the cube lanes faithful to
the FV3 duo-grid Fortran, with no cube-edge artifacts". The short answer is that
**the cases pass their thresholds while two genuine cube defects remain open**,
both of them invisible to the pass/fail gates that scored them.

### Shallow water — 6/6 PASS

| case | status | metric |
|---|---|---|
| `williamson2` | PASS | L2 = 4.68e-04, Linf = 4.44e-03, `v_ll_Linf` = **5.41e-01** |
| `williamson5` | PASS | mass drift 9.71e-16 |
| `williamson6` | PASS | mass drift 1.34e-15 |
| `colliding_modons` | PASS | mass drift 7.28e-16 |
| `cosine_bell` | PASS | L2 = 1.97e-01 |
| `cosine_bell_a0` | PASS | L2 = 1.92e-01 |

The W2 meridional wind on the native cube shows a coherent wave-4 mid-latitude
pattern plus polar speckle, growing from day 0 to day 5.

**This is a known, tuned residual, not a regression.** `v_ll_Linf` *is* this
lane's cube-imprint metric, and `run_atmosphere_test_matrix.py:2895-2915` records
its tuning history: 3.65 → 0.82 m/s (1× hyperdiff) → 0.51 m/s (2×, C36). The
measured 0.541 sits at that documented level.

**Open gap (PLAUSIBLE).** The same comment records **0.38 m/s at C48** for this
production cd-grid lane, against **0.0167 at C48** for the FV3-native six-face
duo stepper — same metric, same resolution, roughly a **23× difference**.
Labelled PLAUSIBLE because the C48 cd-grid figure is quoted from the repo rather
than re-measured here. Hyperdiffusion reduced the imprint; it did not remove it.

### Cube-imprint visual gate — PASS

`scripts/validate/visual_regression.py --check`:

| metric | value | threshold |
|---|---|---|
| SSIM | 1.0000 | ≥ 0.985 |
| per-panel hamming | 0 | ≤ 4 |
| edge-artifact ratio | 1.349 | ref 1.349 |

Verdict line `visual-regression OK`; metric self-test 6 passed, so the gate is
not vacuous. Note this is the W2 v-wind check against the tracked tiny baseline —
sensitive, but narrow.

### Hydrostatic — partial, one real failure

| case | vertical | status | notes |
|---|---|---|---|
| `held_suarez` | sigma | **FAIL** | `DEAD JET max\|v\|=13.0 < 20 m/s (needs ~30; #1028)` |
| `held_suarez` | hybrid | **FAIL** | `DEAD JET max\|v\|=12.6 < 20 m/s` |
| `baroclinic` | sigma | PASS | mass 1.61e-16, max\|v\| = 7.7 |
| `baroclinic` | hybrid | PASS | mass 1.61e-16, max\|v\| = 4.2 |
| `dcmip_transport_11` | sigma | PASS | q1 L2 = 1.39 |
| `dcmip_transport_12` | sigma | PASS | q1 L2 = 0.567 |
| `dcmip_transport_13` | sigma | PASS | — |

**Correction to an earlier verdict in this repo's working notes.** A 3D check
run on 2026-07-28 reported "Held-Suarez cube 3/3 PASS" with `max|v|` of 7.3, 6.9
and 6.9 m/s. That reading was wrong. The `#1028` jet-strength gate
(`run_atmosphere_test_matrix.py:527-535`, added 2026-07-15 in `104a4c682`) cites
**that same 7.3 m/s figure as the dead-circulation value**:

> "Held-Suarez equilibrates at ~30 m/s zonal-mean midlatitude jets; a fully
> spun-up run whose max wind stays far below that has a DEAD circulation, which
> the mass-drift + finiteness gates alone score as PASS. The cd-grid cube ends a
> 200 d run at max|v|=7.3 m/s (#1028) while spectral/latlon/icosahedral reach
> 27-66."

The gate fires only on full runs ≥ 100 days, so the earlier battery never
triggered it and the mass-drift and finiteness gates scored a dead circulation as
a pass. This is the precise failure mode the gate was written to stop, and it
caught the author of the earlier note as well.

`#1028` predates this work by two weeks and HS cube is **not** in
`KNOWN_FAILURES` (only latlon `held_suarez_topo` is, under `#1029`). Jet strength
has improved since the issue was filed (7.3 → 13.0 m/s) but remains far below the
27-66 that spectral, lat-lon and icosahedral reach. **Open circulation defect on
the cube, not a regression from recent work.**

### Non-hydrostatic — pending

Still running at the time of writing. DCMIP TC2/TC3 are expected to fail: that is
the **pre-existing moist-NH cube defect**, discriminator-proven bit-identical on
pre-`#1372` main, root-caused to the corner-divergence vertex halo and currently
blocked behind the raw-slot primitive described below. Not a regression.

### Oracle twins — pending

W2 C48 and case-8 colliding modons against the Zenodo `fv3_solo` build.
Established baselines to compare against: W2 C48 max 0.0167 / RMS 0.0061 inside
the envelope 0.0236 / 0.0096; case-8 day-5 max|V| = 19.75 inside the oracle band
15-27.

## The common blocker

The confirmed cube-vertex divergence defect, the SW imprint gap, and possibly the
weak jet all sit downstream of the cd-grid seam/vertex halo handling. The
velocity halo is fixed and verified but **gated off** (`dgrid_ne_halo=False`),
because on its own it pairs real cross-panel winds with still-edge-replicated
seam metrics and regresses DCMIP TC1 from PASS to a blowup at step 3950.

Completing it requires one primitive that has not been derived from the oracle:
a raw-slot halo `pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg) -> (6, n+2, n+2, 4)`
encoding the eight axis-swap **slot** permutations and the four vertex fills.
`divergence_corner` reads *mixed raw slots* at boundaries — `(j-1,4)+(j,2)` for
`uf`, `(i-1,3)+(i,1)` for `vf` (`sw_core.F90:2187-2207`) — not staggered
`sina`/`cosa`, so the existing pair helper cannot supply it: two inputs and one
common sign cannot determine a four-slot permutation.

## Methodology notes worth keeping

**Three launch errors in this audit produced NaNs and stalls that looked like
model defects and were not.** Recorded because each is cheap to repeat:

1. `--dt 450` is C24-tuned; used at C48 it NaNs at day 1.
2. Reconstructing a documented invocation from a `grep` dropped
   `--ext-bundle --oracle-conventions`, producing another day-1 NaN. **Copy
   cluster-script invocations verbatim.**
3. Three matrix lanes sat `PENDING` for over two hours with reason
   `(PartitionTimeLimit)` — 24 h requested on a partition that caps lower. They
   would never have run. **Check job STATE and REASON, not just that `sbatch`
   returned an ID.**

**A unit-metric test is blind to the metric layer.** The exact stencil
assertions used elsewhere in this work (`+8` seam, `+5` vertex, `+1` sparse) set
`sin=1`, `cos=0`, so they stay green under a wrong raw-slot seam mapping. They
certify the velocity halo and nothing about the metrics; only the solid-body
convergence oracle samples boundary and vertex points with real metrics. Any test
for the raw-slot helper must use **non-unit** metrics.
