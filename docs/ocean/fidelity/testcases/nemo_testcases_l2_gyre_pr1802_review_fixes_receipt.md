# PR #1802 pre-merge review fixes — receipt

**Scope.** Two independent reviewers (codex and a Claude code-reviewer)
converged on one class of defect on branch
`fidelity/nemo-testcases-l2-gyre-codex2`: shared-code changes that alter
behaviour for cards OTHER than the NEMO-literal ones the branch was written
for. Every fix below follows the repo rule (CLAUDE.md RULE 3 / no unasked
choices): **main's behaviour is the default, and the NEMO-literal cards opt in
through an explicit config field.** Where I judge the default SHOULD move, the
default still stays and the move is listed under "Decisions for the operator".

Lane: `phase3/claude_rounds/gyre_pr1802_blockers/repo`, tip `c09a9e111` before
these commits. NEMO source read for every citation:
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/`.

---

## B1 — the NEMO `rmxl_min` arms were collapsed into one

**What was wrong.** `_mixing_length_floor` derived
`rmxl_min = 1e-6/(rn_ediff*SQRT(rn_emin))` for every `tke_mxl_choice` 3/4 card
and ignored `TKEConfig.mxl_min`. NEMO takes that derivation ONLY on the
`ln_zdfiwm = .FALSE.` arm (`zdftke.F90:845-846`). With
`ln_zdfiwm = .TRUE.` it FORCES `rn_emin = 1.e-10_wp` and
`rmxl_min = 1.e-03_wp` (`zdftke.F90:841-843`) and never evaluates it.

Two production cards run `ln_zdfiwm = .TRUE.`:

| card | namelist | branch floor | NEMO floor |
|---|---|---:|---:|
| ORCA1 OMIP (`scripts/run/run_omip_core2.py`, `orca1_zdftke_config`) | ORCA1 deck sets `ln_zdfiwm=T` | 1.0 m | 1.0e-3 m |
| ORCA2-zps testcase card | `cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:396` `ln_zdfiwm = .true.` | 1.0 m | 1.0e-3 m |

Both carry `rn_emin = 1e-10`, so the derivation returns exactly 1.0 m — a
thousand times NEMO's floor, and it silently undid the 2026-08-27 equatorial
undercurrent fix whose reasoning the ORCA1 card still records in its own
comment. Measured on the shipped cards: ORCA1 `iwm_enabled=True` resolves to
`1e-3` after the fix and to `0.9999999999999998` before it; GYRE resolves to
`0.009999999999999998`. The ORCA1 non-iwm arm was wrong too — its own
`rmxl_min = 1e-8` against a derived `1e-2`, a factor of a million.

**What changed.**

* `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py` —
  `TKEConfig.nemo_derived_mxl_min: bool = False`. False (default, main's
  behaviour) = the floor IS `cfg.mxl_min`, which is also the correct value on
  the `ln_zdfiwm = .TRUE.` arm. True = take the derivation.
* `packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py`
  `_mixing_length_floor` — both NEMO arms transcribed and cited; the binary64
  requirement now raises only when the field is True, so no non-NEMO card can
  trip it.
* Cards that opt IN (ln_zdfiwm = .FALSE., derived 1.0e-2 m):
  `fidelity/nemo_recipe.py::_nemo_tke_config` (GYRE and everything built on
  it) and `experiments/dino.py` `DINO_RECIPES["nemo_dino_kamm"]`
  (`tke_nemo_derived_mxl_min`), DINO taking `namelist_ref:1200`'s `.false.`
  default.
* Card that opts OUT with NEMO's forced value:
  `fidelity/nemo_testcase_recipe.py` ORCA2 now sets `mxl_min = 1.0e-3` and
  `nemo_derived_mxl_min = False`, citing `zdftke.F90:841-843`.
* ORCA1 needs no edit: its card already sets `mxl_min = 1.0e-3` from the same
  NEMO lines, and the default now honours it.

**Fails when reverted: YES.**
`tests/ocean/unit/test_nemo_card_opt_in_defaults.py::test_iwm_card_keeps_its_forced_floor_not_the_derivation`
and `::test_orca1_card_resolves_to_the_nemo_forced_floor` both assert 1.0e-3
and both fail against the unconditional derivation (which returns 1.0).
`::test_non_iwm_nemo_card_takes_the_derived_floor` and
`::test_gyre_and_dino_cards_select_the_derived_floor` fail if the opt-in is
dropped from the NEMO cards, which is what pins the GYRE trajectory.

---

## B2 — the carried barotropic seed was selected by state allocation

**What was wrong.** `_carried_nemo_depth_mean` chose NEMO's window seed by
whether `state.uu_b` EXISTED. State allocation is not a scheme selector, and
`experiments/dino.py` handed `nemo_prognostic_barotropic_velocity=True` to
EVERY DINO recipe, so every DINO run silently gained two prognostic leaves and
the NEMO seed identity.

**What changed.**

* `packages/ocean/legoesm/ocean/state.py` —
  `BarotropicConfig.nemo_prognostic_barotropic_state: bool = False`.
* `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` — new
  predicate `nemo_carried_barotropic_state_active(config)`, in the same shape
  as the adjacent `nemo_flux_form_update_active`. `_carried_nemo_depth_mean`
  now takes the config, returns `None` unless the config selects the identity,
  and RAISES when the config selects it and the state has no pair (fail
  closed; never a silent reduction). Both entry points updated.
* NEMO cards select it: `fidelity/nemo_recipe.py::nemo_lat_lon_model_config`
  (one place for every NEMO recipe and for the testcase cards built on it),
  plus the explicit settings and the card gate in
  `fidelity/nemo_testcase_recipe.py`.
* `experiments/dino.py` — `DINOConfig.nemo_prognostic_barotropic_state = False`
  by default and threaded into `BarotropicConfig`; the initial-state builder
  now allocates the pair only when the card asks. `nemo_dino_kamm` (and, by
  spread, `nemo_dino_kamm_mlf`) sets it True — the lane's DINO gates (the
  169/170 card tests and the round-184 developed-state bridge) were passing
  with the pair, so that arm is unchanged. Every other DINO recipe
  (`legoesm_default`, `nemo_paper`, `veros`, `mitgcm`, `oceananigans`) carries
  no new prognostic state, which is main's behaviour.

**B2(ii) — the `U_bar_corr` recomputation is NOT threaded, deliberately.** The
replacement depth-mean at `barotropic_substeps_latlon_cgrid` keeps the shared
min-rule / generic reduction even on a card whose window SEED uses
`nemo_ssh_avg` + `nemo_literal`. Threading the card's convention there is not a
keyword change: the literal evaluation is a different code path needing
`eta_dyn`/`H_bathy`/`area`/`z_coord`, and it would change every certified NEMO
trajectory (GYRE, ORCA2, DINO). Main has the same asymmetry on its own
`_seed_override` path, so "default stays" means leaving it. It is recorded as a
named, cited comment at the call site rather than left silent, and it is
decision D4 below.

**Fails when reverted: YES.**
`test_nemo_card_opt_in_defaults.py::test_carried_seed_is_selected_by_config_not_by_state_presence`
asserts that a state WITH the pair and a config WITHOUT the flag still reduces
— which is exactly what the state-presence branch could not do — and that the
flag without a pair raises.
`::test_non_nemo_dino_recipes_allocate_no_new_prognostic_state` fails if the
unconditional `True` comes back.

---

## B3 — file-backed tripole construction (two separate answers)

### B3(a) mesh `ff_t` — CONVENTION CHANGE, now opt-in

NEMO reads `ff_t`/`ff_f` from `cn_domcfg` whenever BOTH exist
(`domhgr.F90:222-227`); otherwise it recomputes them. The branch adopted the
file's `ff_t` for `f_T` unconditionally, which moves every file-backed tripole
run (ORCA1 OMIP production included) in the last bits.

`packages/core/legoesm/grids/tripole.py::create_tripole_grid` gains
`use_mesh_coriolis: bool = False`. Default = the analytic
`2*Omega*sin(lat_T)`, i.e. main. The ORCA2 NEMO card
(`fidelity/nemo_testcase_recipe.py`) opts in, citing `domhgr.F90:222-227`.
`ff_f` keeps being carried either way: it is a pure addition, read only by the
literal NEMO EEN/ENE vorticity arms.

**Fails when reverted: YES.**
`tests/grids/test_tripole_internals.py::TestMeshCoriolisIsOptIn::test_default_keeps_the_analytic_coriolis`
writes a mesh whose `ff_t` is a constant unrelated to latitude and fails the
moment the read becomes unconditional;
`::test_opt_in_reads_the_mesh_field` fails if the option is dropped.

### B3(b) U-point angle padding — GENUINE LAYOUT DEFECT, keep the fix

**Determined from the code, not assumed.** legoESM's u-face `j` is the WEST
face of cell `j`: `interp_cell_to_uface` pads the lon halo and averages
`f_pad[j], f_pad[j+1]`, so face 0 is `mean(f[-1], f[0])`. The tripole Coriolis
build already matched that convention
(`f_u_inner = 0.5*(roll(f_T,1,axis=1) + f_T)`). NEMO's u-point `i` is EAST of
T-cell `i`, so `e1u[i]` and the `_compute_rotation_angles` output at column `i`
both belong at face `i+1`, with the last column wrapping into face 0. Main
APPENDED the first column for the metrics AND the angles, i.e. it handed every
face the operand of the face one column east. That is a real layout defect on a
curvilinear mesh (invisible on a uniform one), and the branch's prepend is the
fix. The branch had already corrected and tested the METRICS; the ANGLES were
corrected without a test, which is what the review caught.

**Kept as a bug fix, now tested.**
`tests/grids/test_tripole_internals.py::TestUPointMetricAlignment::test_u_rotation_angles_pad_like_the_u_metrics`
recomputes the raw angles, asserts a non-vacuity condition (the angle really
varies along `i`), then pins `cos/sin_alpha_u[:, 1:] == raw` and
`[:, 0] == raw[:, -1]`, and asserts it is NOT `raw[:, 0]` — the old layout.
**Fails when reverted: YES** (last assertion).

---

## S4 — the literal NEMO EEN/ENE Coriolis now fails closed

`nemo_een_ene_vertex_coriolis` fell back to the generic V-point field whenever
`grid.ff_f` was absent. On a RECTILINEAR grid that fallback is exact — V and F
share a latitude, so the generic value IS the F-point value — and DINO's
beta-plane `een_total` card depends on it. On a CURVILINEAR grid they are
different staggerings, so the fallback substitutes a different operand. The
function now raises on a curvilinear grid (active tripolar fold) with no
`ff_f`, naming the missing operand, and keeps the exact fallback elsewhere.

**Fails when reverted: YES.**
`test_nemo_card_opt_in_defaults.py::test_literal_een_coriolis_refuses_a_curvilinear_grid_without_ff_f`
(both arms: equality on the rectilinear grid, raise on the folded one).

---

## S5 — `eice` renumbering reached only half the code

Decision 46 (user-approved) renumbered `TKEConfig.eice`/`KPPConfig.eice` onto
NEMO's `nn_eice`: 1 = `1 - TANH(fr_i*10)`, 2 = `1 - fr_i` (the OLD meaning of
1), 3 = `1 - MIN(1, 4*fr_i)` — `zdftke.F90:246,253-258`, validated against
`zdftke.F90:828-835`. `physics/vertical_mixing/fesom_integration.py` still
implemented the OLD meaning for 1 and REJECTED 2, so one value had two
meanings depending on which integration ran. It now routes through the shared
`nemo_tke_effective_ice_fraction` and accepts 0/1/2/3.

**Fails when reverted: YES.**
`::test_fesom_eice_uses_the_shared_nemo_numbering` asserts mode 1 is
`tanh(10*fi)`, that modes 1 and 2 differ, and that the old inline
`_eice == 1 else` expression is gone; `::test_fesom_accepts_nn_eice_2` pins the
widened guard.

---

## S6 — the `ln_mxl0` anchor's `tmask` operand became a hard error

`_mxl0_surface_anchor` raised whenever `surface_tmask` was None for
`tke_mxl_choice` 3/4. `fesom_integration.py` never passes one, so a FESOM card
on a NEMO mixing-length choice could no longer run — a previously tolerated
condition turned into an abort without being asked.

`TKEConfig.nemo_mxl0_surface_tmask: bool = False`. False (default, main's
behaviour) = the unmasked stress, mask ignored, nothing raises. True = the
compiled `taum(:,:)*tmask(:,:,1)` statement (`zdftke.F90:640-642`) with a
missing mask a hard error. The NEMO cards (`_nemo_tke_config`, DINO
`nemo_dino_kamm`) set True. The lat-lon C-grid driver supplies the mask for
every choice-3/4 card, so no NEMO card loses it.

**Fails when reverted: YES.**
`::test_anchor_without_a_surface_mask_is_accepted_by_default` fails against the
unconditional raise; `::test_nemo_card_requires_the_masked_statement` fails if
the opt-in arm stops masking or stops raising.

---

## S7 — two branch-only inline coefficients (CI ratchet)

* `physics/bbl_adv.py` `active3 = h > 1.0e-3` — a wet-cell thickness floor,
  identical to the three sibling sites in the same file that already carry the
  marker. Given the same real `# coeff-ok: wet-cell thickness floor [m]`.
* `physics/shortwave_penetration.py` — the RGB class index now reads the
  module-level `_RGB_CLASS_INDEX_OFFSET` / `_RGB_CLASS_INDEX_SLOPE` that were
  already declared with their provenance comment (NEMO `trc_oce.F90`;
  consumed by `traqsr.F90`'s `qsr_RGBc`), instead of repeating 41.0 / 20.0.

Ratchet result and the two PRE-EXISTING main failures are recorded under
"Gate results" — `fesom_integration.py` and
`cg_helmholtz_mixed_precision_1675.py` are not mine and are left alone.

---

## S8 — a restart migration that could never succeed

`_migrate_v3_deviation_bt_hist` reconstructed the absolute AB3/AM4 histories
from `uu_b`/`vv_b`, which are version-4 slots: no version-3 archive holds them,
so the function always reached its own `fail()`. Reconstructing from the 3-D
depth mean needs the grid, the face masks and the live layer thicknesses, none
of which the loader has — not achievable in the 60-line budget, and a wrong
reconstruction here silently reinterprets deviations as absolute histories.

Replaced by `_refuse_v3_deviation_bt_hist`: a version-3 archive that carries
`bt_hist` is refused in plain words ("must be REGENERATED with the current
build"); a version-3 archive WITHOUT `bt_hist` — every run that did not use the
NEMO AB3/AM4 barotropic filter — still loads unchanged, so format 3 stays
readable.

**Fails when reverted: YES.**
`::test_v3_bt_hist_archive_is_refused_with_a_readable_message` pins both arms
(pass-through with no history, plain refusal with one).

---

## N9 / N10 / N11 (notes)

* **N9** `timestepping/tridiagonal.py` — the docstring promised the `_TINY`
  pivot clamp unconditionally while the `nemo_unnormalised` branch has none.
  The docstring now states both arms and why the unclamped one is safe: its
  only caller, the sea-ice vertical heat solve (`ice/bitz_lipscomb.py`), builds
  a diagonally dominant matrix, and a new caller must establish the same
  property. Docstring only; no behaviour change.
* **N10** `fidelity/provenance.py` — the two `_git` closures are folded into
  one module-level `_git_output(tree, *args)`; `git_sha` keeps its own error
  text by wrapping the two calls in one `try`. No behaviour change.
* **N11** `mpas_config.py` `K_zeta_bih_ref_dx_m` 120194.609375 →
  120194.60581296285 with the float64 mesh mean — KEPT, it is the correct
  anchor. Listed under "Behaviour changes" because it moves MPAS runs that do
  not pin `K_zeta_bih` by ~1e-7 relative.

---

## Behaviour changes for the PR body

1. **`eice` renumbering (decision 46).** `TKEConfig.eice` / `KPPConfig.eice`
   now follow NEMO `nn_eice`: 1 = `1 - tanh(10*fr_i)`, **2 = `1 - fr_i`, which
   is what 1 used to mean**, 3 = `1 - min(1, 4*fr_i)`. Any saved card or tuned
   JSON carrying `eice=1` with the old intent must be moved to `eice=2`. As of
   this commit FESOM obeys the same numbering as the lat-lon and MPAS paths;
   before it did not.
2. **MPAS `K_zeta_bih` anchor.** `K_zeta_bih_ref_dx_m` is now
   120194.60581296285 and the mesh spacing mean is accumulated in float64.
   MPAS runs that do not pin `K_zeta_bih` change by ~1e-7 relative.
3. **ORCA2-zps mixing-length floor.** 1.0e-3 m (NEMO's `ln_zdfiwm` forced
   value) instead of the branch's derived 1.0 m and main's 1.0e-8 m. The ORCA2
   card must be re-certified; see decision D1.
4. **ORCA1 `ln_mxl0` anchor is masked.** The ORCA1 OMIP card now evaluates
   the anchor on `taum*tmask(:,:,1)`, which differs from the previous unmasked
   form on land columns only.
5. **NEMO `ln_mxl0` anchor floor.** For `tke_mxl_choice` 3/4 the anchor floor
   is now `rmxl_min` (NEMO overwrites `rn_mxl0` with it at
   `zdftke.F90:859-862`) rather than the deleted `TKEConfig.mxl0_min_m`
   (default 0.04 m). Only choice-3/4 cards are affected, i.e. only NEMO cards.
6. **DINO prognostic barotropic state.** The NEMO DINO cards carry
   `uu_b`/`vv_b`; every other DINO recipe no longer does. The DINO Y5
   certification predates the prognostic pair and must be re-run against the
   NEMO DINO cards — see decision D2. Restart archives written by a NEMO DINO
   run therefore hold two extra prognostic slots.
7. **File-backed tripole grids.** `ff_f` is now carried on the geometry (read
   only by the literal NEMO EEN/ENE arms); `f_T` is unchanged unless a card
   passes `use_mesh_coriolis=True` (only the ORCA2 card does). The U-point
   metrics AND rotation angles now both take the face-to-their-west operand:
   a layout fix that changes every file-backed tripole run on a NON-uniform
   mesh, including ORCA1 OMIP production.
8. **Restart format 3.** An archive carrying the deviation-form barotropic
   history is now refused with an explicit message instead of failing inside a
   migration that could never succeed. Format-3 archives without that history
   are unaffected.

---

## Decisions for the operator

Each row is "default stays (main) vs moves (NEMO)". The default stays in every
case; these are the moves I recommend, and what they would touch.

**D1 — ORCA2-zps mixing-length floor. ALREADY MOVED, flagged.** This is the one
place I did not leave the default alone, because all three candidate values
disagree: main 1.0e-8 m, branch 1.0e-3 m via `cfg.mxl_min` only after this fix,
branch-before-fix 1.0 m, NEMO 1.0e-3 m. A card whose entire contract is NEMO
literalness cannot ship a floor a thousand times NEMO's, and the ORCA2 deck's
`ln_zdfiwm = .true.` is in its own namelist at
`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:396`. **Recommend: keep the move.**
Affects: the ORCA2-zps card and every ORCA2 lane gate; they need re-running.
Revert is one line (`mxl_min=1.0e-3` → drop, `nemo_derived_mxl_min=True`).

**D2 — DINO Y5 certification.** The NEMO DINO cards keep the prognostic
external mode (the lane's DINO gates pass with it), but the DINO Y5
certification predates it. **Recommend: re-run DINO Y5 on the NEMO DINO cards
before citing it again.** Nothing to change in code.

**D3 — ORCA1 `ln_mxl0` surface mask. MOVED after review, flagged.** Codex
raised it as a HIGH inconsistency: the ORCA1 card is NEMO-literal, runs
`ln_mxl0 = .TRUE.`, and was the only such card left unmasked. It now sets
`nemo_mxl0_surface_tmask=True` (and `nemo_derived_mxl_min=False` explicitly,
since both of its arms already carry NEMO's own `rmxl_min`). The mask is
`tmask(:,:,1)`, which is 1 on every WET column, so masked and unmasked differ
only on LAND columns — the anchor there collapses to the floor. **Recommend:
keep the move.** Affects: ORCA1 OMIP, land columns only. Revert is one line.

**D4 — the barotropic `U_bar_corr` convention.** The replacement depth-mean
keeps the generic min-rule reduction on cards whose window seed is
`nemo_ssh_avg` + `nemo_literal`. **Recommend: move it, but as its own round
with its own re-certification**, because it changes GYRE, ORCA2 and DINO
trajectories and needs the literal evaluation's operand set threaded through.
Affects: every NEMO card. Left as a cited comment at the call site.

**D5 — `use_mesh_coriolis` on other tripole cards.** Only the ORCA2 card opts
in. **Recommend: leave the rest analytic** until someone measures what the
ORCA1 mesh's stored `ff_t` does to a 20-year OMIP run; the difference is
last-bits but it is not zero.

---

## Review

Both reviews ran on this diff, before it was declared done.

**codex** (`codex exec --sandbox read-only`) returned **HOLD** with six
findings. Five are closed in the follow-up commit:

1. HIGH, ORCA1 missed the `ln_mxl0` mask opt-in — FIXED, see decision D3.
2. HIGH, `tests/ocean/unit/test_tke_nemo_terms.py` (a push-gate test) still
   expected the unconditional derivation and the unconditional raise — FIXED:
   the test now states the card's two selections and additionally pins the
   default arm (configured floor, missing mask accepted).
3. MEDIUM, `scripts/validate/ocean_fidelity/frozen_column_tke_twin.py` passed
   four arguments to `_mxl0_surface_anchor` — FIXED by giving `surface_tmask`
   a `None` default, which is also what the unmasked arm wants.
4. HIGH, "B3b is not established and is likely mis-indexed" — **NOT ACCEPTED as
   stated, and split.** Codex is right that legoESM's angle DEFINITION is not
   NEMO's: `_compute_rotation_angles` takes a forward difference of
   `glamu`/`gphiu` along i, while NEMO's `gsinu`/`gcosu` use the vector between
   the F points below and above that U point (`geo2ocean.F90:168,259-260`).
   That difference predates this branch and is unchanged by it. The question
   B3b actually asks is narrower: the array is shaped like `gphiu` and its
   element `i` is built from `gphiu[:, i]`, so it is indexed by NATIVE U point
   `i`, exactly as `e1u` is — and the metrics were already corrected and tested
   to sit at face `i+1`. Leaving the angles on the old append would put the two
   operands of one rotation on different faces. Registered as an open, separate
   item: legoESM's U/V rotation angles are a forward-difference estimate, not
   NEMO's F-point construction.
5. MEDIUM, `use_mesh_coriolis` did not implement NEMO's PAIRED rule — FIXED:
   `domhgr.F90:222-223` guards `ff_f` and `ff_t` with one `.AND.`, so a mesh
   holding only one of them now supplies neither, and `ff_f` is carried only
   with its partner. Tested.
6. LOW, several new tests were textual — FIXED: the FESOM test now BUILDS the
   profiles function for every `nn_eice` value (2 raised before the change) and
   checks an unknown value still raises; the DINO test now resolves the two
   recipes' configs and builds a `legoesm_default` state to show it carries no
   `uu_b`/`vv_b`.

**Claude code-reviewer** (independent, fresh context) returned **no blockers**:
no certified NEMO card silently moved, no leak in the other direction, no stale
callers of the three changed/renamed symbols, no vacuous tests, and it
independently CONFIRMED the B3b west-face convention from
`interp_cell_to_uface` and the measured `glamu - glamt = +0.5` deg. It
separately surfaced the ORCA1 mask gap as the receipt's own decision D3, which
is the same item as codex finding 1.

The two reviewers disagreed on exactly one point, codex finding 4 versus the
Claude reviewer's confirmation of B3b. The discriminating check is a grep, and
it was run: the padding question and the angle-definition question are
different questions, and the answer above separates them.

## Gate results
