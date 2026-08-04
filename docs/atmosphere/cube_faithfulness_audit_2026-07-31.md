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

The confirmed cube-vertex divergence defect and the SW imprint gap sit
downstream of the cd-grid seam/vertex halo handling.

**RETRACTED (2026-08-01): the dead Held-Suarez jet is NOT downstream of these.**
An earlier version of this section speculated that all three shared a cause. That
was checked and refuted on the code paths, under default matrix settings:

- The corner-divergence path is entered only when corner-divergence damping is
  active (`primitive_eq_cdgrid.py:601`), whose gate requires nonzero CDD
  selectors (`_fv3_divergence_corner.py:60`); the matrix defaults
  `LEGOESM_CDD_D2BG`, `D4BG` and `NORD` to zero
  (`run_atmosphere_test_matrix.py:4226`). So the `dgrid_ne_halo=False` branch and
  the missing west ghost **cannot contribute to standard dry Held-Suarez at all**.
- The raw-slot `d2a2c_vect` path is reached in PE only by the flux-form moisture
  substep (`primitive_eq_cdgrid.py:1484`, `:1529`), which needs
  `moisture_flux_form=True` and tracers (`:1986`); the default is false and dry
  HS initialises no tracers (`held_suarez.py:296`).

Their direct contribution to this lane is zero unless a run sets non-default CDD
or moisture/tracer options. The 23× W2 imprint is **not transferable evidence** of
a Held-Suarez momentum sink. Broader non-duogrid seam paths remain plausible, but
that is a separate investigation with its own evidence bar.

The
velocity halo is fixed and verified but **gated off** (`dgrid_ne_halo=False`),
because on its own it pairs real cross-panel winds with still-edge-replicated
seam metrics and regresses DCMIP TC1 from PASS to a blowup at step 3950.

Completing it requires one primitive, now DERIVED from the oracle (see the appendix):
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

## Appendix: the raw-slot seam mapping, derived

The primitive that blocks the corner-divergence port, derived from the oracle
(round 5). Recorded here because it was previously only in a scratch file.
Slots are `(W, S, E, N)` = indices `(0, 1, 2, 3)`.

**Eight quarter-turn (axis-swapping) seams** — `sin` sign `+`, `cos` sign `−`:

| destination ghost seam | neighbour edge | reverse | destination `(W,S,E,N)` gets source slots |
|---|---|---|---|
| `1:N` | `4:E` | no | `(S,E,N,W)` = `(1,2,3,0)` |
| `4:E` | `1:N` | no | `(N,W,S,E)` = `(3,0,1,2)` |
| `1:S` | `5:E` | yes | `(N,W,S,E)` |
| `5:E` | `1:S` | yes | `(S,E,N,W)` |
| `3:N` | `4:W` | yes | `(N,W,S,E)` |
| `4:W` | `3:N` | yes | `(S,E,N,W)` |
| `3:S` | `5:W` | no | `(S,E,N,W)` |
| `5:W` | `3:S` | no | `(N,W,S,E)` |

**Four half-turn seams** — destination `(W,S,E,N)` ← source `(E,N,W,S)` =
`(2,3,0,1)`, reversed, `sin` and `cos` signs both `+`:
`(2,S)←(5,S)`, `(2,N)←(4,N)`, `(4,N)←(2,N)`, `(5,S)←(2,S)`.

The remaining twelve directed seams are identity-slot, positive-sign copies.

**Four vertex fills**, applied after all side strips, identical for `sin` and
`cos` with no sign change (`L = n+1`, padded coordinates):

```text
SW: p[:,0,0,E] = p[:,0,1,S]   ; p[:,0,0,N] = p[:,1,0,W]
SE: p[:,L,0,W] = p[:,L,1,S]   ; p[:,L,0,N] = p[:,L-1,0,E]
NE: p[:,L,L,W] = p[:,L,L-1,N] ; p[:,L,L,S] = p[:,L-1,L,E]
NW: p[:,0,L,E] = p[:,0,L-1,N] ; p[:,0,L,S] = p[:,1,L,W]
```

This is the post-`fill_ghost` repair at `fv_grid_utils.F90:580` (SW/NW) and
`:609` (SE/NE). **The other two slots in each diagonal ghost cell must remain
invalid** — FV3 deliberately leaves them at `tiny_number = 1e-8` for `sin` and
`big_number = 1e8` for `cos` (`fv_grid_utils.F90:51`). Do not invent values for
them.

The helper needs **its own cached table**: `_get_axis_swap_tables_h1(n)` encodes
staggered `u(n,n+1)`/`v(n+1,n)` locations and *velocity-component* signs, whereas
raw slots are square-cell data with a slot-channel permutation. The eight rows
and reversal flags there are the right topology source (`dgrid_halo.py:284`) —
its velocity signs are not.

### Caveats and scope, stated by the deriver

- **Face-ID correspondence is not independently locked.** The supplied FV3 source
  does not include FMS's mosaic contact table, so face IDs come from this repo's
  canonical gnomonic `CONNECTIVITY`. The permutation and sign derivation is not a
  guess, but to lock the face-ID mapping independently, dump slots 1:4 immediately
  after FV3's special repairs on a stretched C5 run and compare every side ghost
  against the table above.
- **This helper alone does not justify enabling `dgrid_ne_halo`.** It fixes the
  raw-metric inconsistency; `ua/va` and the paired staggered metrics remain
  separate blockers.
- **It does not by itself touch the SW imprint or the dead jet.** There is a real,
  testable connection to the imprint: production `c_sw`/`d2a2c` pad the W/S/E/N
  `sin_sg` slots independently (`fv3_sw_core.py:299`, `:1369`) — the same wrong
  abstraction — but those callers use interpolation offsets, so this copy-only
  helper cannot be substituted mechanically; an offsets-aware version plus a C48
  A/B is needed. The dead-jet link is **only a hypothesis** until that A/B shows a
  seam-tendency or momentum-budget change.

### Test requirement

The test must use **non-unit metrics** and must go red for identity-slot copying,
a wrong permutation, a wrong reversal, or a missing `cos` sign — none of which the
existing `sin=1, cos=0` assertions can detect. Assert all eight vertex assignments
and the two poisoned slots per diagonal cell. The real-metric solid-body
convergence test remains the end-to-end gate.

## Held-Suarez dead jet — investigation ladder (2026-08-01)

Cheapest-first, with the rule that a new 200-day run is **not** justified until
the offline diagnostics show a difference:

1. **Offline C↔D wind-transfer measurement** on saved state.
2. **One-state term / angular-momentum / seam budget.**
3. A **5-10 day persistent-D versus default pair** — only if 1-2 show a
   difference.

Interpretation, decided in advance so the result is not read after the fact:

| early result | most likely direction |
|---|---|
| `T'` grows but EKE collapses | C↔D wind projection |
| EKE grows but `u'v'` convergence is wrong | seam / metric / momentum route |
| EKE and `v'T'` both fail | PE baroclinic conversion, or broad dissipation |
| correct fluxes but weak mean acceleration | drag / AAM / mean-flow coupling |

**Instrument warning.** `scripts/run/run_dry_held_suarez_cube.py:109` computes
`eddy_ke` by subtracting **face-local** components rather than geographic winds.
That diagnostic must be recomputed before it is used physically — on the cube,
face-local and geographic eddy fields are different quantities.

## Claims tightened (2026-08-02, external review)

The three headline results above were stated too strongly. Each is real, but each
certifies less than the wording implied.

**The W2 twin is an EXTERNAL, not a controlled, comparison.** Our runner samples
through a `c2l-ord2` lens with nearest-cell remapping
(`run_duo_stepper_w2.py:7`), while the Zenodo reference is regridded with
`fregrid`. The gate itself says so (`w2_duo_oracle_gate.py:12`). So `0.0167`
reproducing the baseline exactly is faithful **for the declared bounded-duo
configuration and that observable** — it is not a like-for-like field comparison
against the Fortran.

**Equal modon peaks establish symmetry and amplitude only.** `peakW == peakE` to
every digit rules out cube-induced asymmetry in the peak, and day-5 `19.753` vs
`19.75` matches the amplitude baseline. Neither statement constrains **phase,
trajectory, or field fidelity**. The stronger check is a native-lattice
checkpoint comparison; the modon runner already supports state dumps
(`run_duo_stepper_modon.py:63`).

**The visual gate is a regression fingerprint, not a physical oracle.** It
generates its baseline from the same model (`visual_regression.py:137`, `:182`),
so `SSIM 1.0000` means "unchanged since the reference was taken", not "physically
correct". CI runs it non-blocking (`ci.yml:223`).

None of this retracts a measurement. It bounds what the measurements support.

## Ranked improvements (external review, by expected error reduction)

1. **Make a recipe the executable source of truth.** One checked-in recipe fixes
   case, grid, nlev, vertical coordinate, dt policy, days, oracle mode, required
   flags, artifact schema, reference digest, commit and scheduler profile; a
   single submitter resolves, validates, submits and writes `run.json`, and
   `.sbatch` files become thin executors. Gates must **reject** an artifact whose
   embedded recipe hash, `n`, `dt`, `ext_bundle`, `oracle_conventions`,
   vector-corner mode or reference digest disagree with the request — turning a
   dropped flag into a contract failure instead of an interpretive mistake. This
   alone removes nearly all six launch errors made during this audit;
   `duo_stepper_w2.sbatch:16` can currently pass `--ext-bundle` without
   `--oracle-conventions`, and `w2_duo_oracle_gate.py:69,:149` trusts a
   caller-supplied `--res`.
2. **Separate model-state diagnosis from generic-driver control**, via
   matrix-native day-10 checkpoints, and replace the Held-Suarez health quantity
   with a named physical diagnostic — area-weighted zonal-mean geographic
   eastward jet at a stated latitude/level, plus EKE and the latitude/level of
   the peak. The present gate compares a **pointwise local wind magnitude**
   against prose about a ~30 m/s **zonal-mean** jet
   (`run_atmosphere_test_matrix.py:527`, `:4299`). The dead jet is real; the
   comparison is not like-for-like.
3. **Put the cube-fidelity tests in required CI and test the merge result.** CI
   collects everything but executes only selected grid files (`ci.yml:74`,
   `:169`) — the raw-slot, metric-pair, W2 gate and D-grid regression tests are
   not among them. Add a required `cube-fidelity` job, add `merge_group`
   triggers, require checks on the merge SHA, and dismiss stale approvals. The
   stale-head merge that dropped this campaign's review fixes is exactly what
   that prevents.
4. **Replace the remaining blind and common-mode tests.** Named as highest risk:
   `test_dgrid_metric_pair_halo.py:48` (sign test compares the helper to itself;
   uniform field cannot expose orientation), `test_dgrid_vector_halo_iter1078.py:12`
   and `test_dgrid_halo_iter1076.py:30` (spatially constant faces make reversal
   invisible; the latter explicitly accepts axis-swap edge fallback),
   `test_d2a2c_ua_va_halo.py:49` (its reference is the same implementation, and
   its "not a four-corner average" control uses unrelated random corners), and
   the `iter326`/`iter335`/`iter234` imprint A/Bs, which prove wiring changes
   something rather than that either branch is right. The raw-slot canonical
   table from #1445 is the pattern to copy. Residual gap: face IDs still come
   from repo `CONNECTIVITY` (`dgrid_halo.py:382`) — one stretched-C5 FV3 slot
   dump would close it as a fixture.
5. **Do not make the corner-divergence port the next investment.** It is real
   fidelity debt and eventually necessary for NH, but it is gated off, the
   standard HS CDD settings default to zero (`run_atmosphere_test_matrix.py:4111`),
   and it cannot explain the dead jet. When it is done, do it as **one vertical
   slice** — raw slots, paired metrics and `d2a2c` wired together, validated
   against an independent fixture, real-metric solid body XPASSing, TC1 restored
   — rather than accruing more helper-only work.
