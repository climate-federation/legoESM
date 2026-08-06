# Round-1 finding classification + disposition

All 7 confirmed bugs accepted; 4 root-caused to ONE design flaw (once-daily
86400 s snapshot advance) and fixed by advancing the skin per MODEL step.

| # | Finding | Class | Disposition |
|---|---------|-------|-------------|
| 1 | Instantaneous flux integrated as a whole day (diurnal aliasing, 2.72 K/day) | CONFIRMED | FIXED — advance per model step (dt=DT) with the fresh per-step `_sfc_diag`; the diurnal SW/turbulent cycle is now resolved at the radiation-update cadence. |
| 2 | Restart drops the pending daily advance (1.07 K/restart) | CONFIRMED | FIXED — per-step advance means the checkpointed `ice_T_skin` is ALWAYS fully advanced through the last step; no pending daily advance to drop/double-count. New `test_restart_split_invariance` proves the pure-carry property. |
| 3 | CMOR `tas` extrapolates from constant `T_ice`, not the skin | CONFIRMED | FIXED — pass `self._ice_T_skin` (feature on) to `_tas_2m` at the CMOR feed. |
| 4 | Melt cap at basal `T_freeze_ocean` (271.35) not surface `T_freeze` (273.15) | CONFIRMED | FIXED — new `T_melt_surface_K=constants.T_freeze` cap; base + open-water snap keep `T_freeze_ocean`. Mirrors `ice/sea_ice.py` `T_base` vs `T_melt_surface`. Tests updated. |
| 5 | SIC applied on the wrong side of the interval | CONFIRMED (minor) | ADDRESSED by the per-step redesign: the skin advances with the current step's flux and the current day's SIC; an open→ice cell was snapped to the freezing point while open (`where(sic>0)`) and begins cooling from there. |
| 6 | Save-after-load / feature-on→off reuse can launder a stale skin | CONFIRMED (narrow) | FIXED — checkpoint save gated on `self.config.mpas_ice_skin_prognostic`. |
| 7 | "Unconditionally stable" is only conditional | CONFIRMED (clarification) | FIXED — per-step dt=DT keeps `r*lambda = DT*lambda/C << 1` (stable, monotone for realistic lambda); docstring now states the coupled system is conditionally stable and that the cap bounds (does not cure) divergence. |

Limitations (round-1 "minor"):
- Partial-SIC single-tile closure error — documented in the helper docstring
  (exact only at sic=1; second-order via the sic-weighted anchor).
- `blend_surface_temperature` type contract — signature/docstring now
  `T_ice: float | array`.
- Differentiability caveat — documented (interior smooth `r/(1+r*g)`; the
  clip/mask dead-gradient is the physically correct saturated-cap / inactive-
  mask behaviour, not a defect).
- Global `h_ice` — documented as a single global climatological thickness.

Residual (declared, not fixed): fresh-run spin-up (~2 months warm-biased toward
the old behaviour) — inherent to the T_f seed; declared in the setup log +
comment; scorecard uses months 3-12. Restart bit-identity inherits the model's
existing first-step-re-solves-radiation ~1e-8 non-determinism (pre-existing, not
introduced by the skin).

---

# Round-2 finding disposition

| Codex-2 | Finding | Class | Disposition |
|---------|---------|-------|-------------|
| 1 (NEW) | Mid-day restart re-blends the anchor from the already-advanced restored skin -> O(0.1-1 K) surface-boundary divergence vs the straight run | CONFIRMED | FIXED — the anchor T_sfc is now re-blended EVERY model step from the cached daily SST/SIC against the current skin (`_blend_T_sfc`). Straight and restart both anchor on the same restored skin at the resume step, so no branch beyond the pre-existing radiation-first-step recompute. |
| 2 (#7) | Feedback still daily-lagged (physics consumes the day-held anchor), so the DT*lambda/C per-step stability claim did not apply | CONFIRMED | FIXED by the SAME per-step re-anchor: the physics now consumes the freshly advanced skin each step, so `r*lambda = DT*lambda/C << 1` genuinely holds. Helper docstring restated: stability requires advance and anchor-refresh cadence to MATCH. |
| 3 (#6) | load->save-without-run still strips the skin (save reads the live field which load cleared) | CONFIRMED | FIXED — save now falls back to the staged `_carry_aux["ice_T_skin"]` when the live field is not yet adopted, so a save before the run persists it. Still config-gated (no off-feature leak). |

Codex-2 confirmed FIXED and NOT re-flagged: #1 (flux-time), #3 (tas), #4 (melt
cap), #5 (SIC side). Perf note (wrap the per-step update in jax.jit) — declined:
the update is a handful of elementwise ops on (nCells,), eager like the
sibling per-step qv-smooth / hard-sat drains; a jit wrapper adds a
dispatch/compile with no measurable gain at this op count.

Added (codex-2 asked for a feature-on restart test): end-to-end subprocess test
`test_end_to_end_mpas_ice_skin_wiring_and_checkpoint` — real `_run_mpas`
per-step advance + per-step T_sfc re-anchor + checkpoint save + restart from the
day-1 checkpoint (load->stage->adopt), skin reproduced. Cooling magnitude not
asserted (analytical SST/SIC is ice-free, sic=0; helper equilibrium tests cover
the cooling physics). Full mid-day-split bit-diff deferred: `--checkpoint-days`
is integer so the CLI cannot checkpoint mid-day; the consistency is guaranteed
by the per-step re-anchor construction + `test_restart_split_invariance`.

---

# Round-3 finding disposition

| Codex-3 | Finding | Class | Disposition |
|---------|---------|-------|-------------|
| NEW | Feature-on shape guard bypassable: with an (nCells,) skin, a scalar/mis-counted SST/SIC broadcasts to (nCells,) and passes the output-only check | CONFIRMED | FIXED — the setup guard now validates the RAW SST and SIC shapes (`reshape(-1).shape == (nCells,)`) before blending, restoring the strength the old scalar-T_ice blend had. |
| defensive | Checkpoint skin only size-checked; a NaN poisons even sic=0 anchors via 0*NaN in the blend | CONFIRMED | FIXED — the seed overlay refuses a non-finite restored skin (`jnp.all(jnp.isfinite)`); new subprocess test `test_restart_refuses_nonfinite_checkpoint_skin` corrupts a checkpoint and asserts the resume raises. |
| obs | Radiation-subcycle restart branch now enters the skin update (restart step 0 re-solves while the straight run may hold SW/LW) | ACKNOWLEDGED | Not a new bug — codex agrees it is the PRE-EXISTING radiation restart branch (state-wide), not a stale-anchor branch. Declared in rebuttals/report; the skin inherits the model's existing restart tolerance, it does not amplify it. |
| #3 | load->save-without-run: save fallback safe, but physstate_* staging raises first | RESOLVED | Codex confirms NO silent strip remains; the raise is pre-existing behaviour for an unsupported op (load->save with no run). |
| cleanup | Redundant jnp.asarray in the _blend_T_sfc split | DONE | `_compute_T_sfc` now passes raw SST/SIC (blend asarrays once); output byte-identical. |
| perf | Don't claim the per-step re-blend is free | NOTED | It is an eager O(nCells) unfused blend+lapse each step, of the same class as the sibling per-step qv-smooth/hard-sat drains; negligible vs the MPAS dynamics but not claimed free. |

---

# Round-4 finding disposition

| Codex-4 | Finding | Class | Disposition |
|---------|---------|-------|-------------|
| 1 (edge) | Feature-on `(nCells,1)` SST/SIC broadcasts to `(nCells,nCells)` against the `(nCells,)` skin and is rejected (the scalar-T_ice blend tolerated it via a trailing reshape) | CONFIRMED | FIXED — `_blend_T_sfc` flattens SST/SIC to `(nCells,)` BEFORE blending, so a 2-D source blends elementwise. New unit test `test_blend_flattens_2d_source_against_array_skin` (elementwise result + documents the un-flattened `(n,n)` bug). |

Codex-4 confirmed item 2 (NaN refusal) fully fixed and the raw-shape guard
correct for scalar/mis-counted and `(nCells,)` inputs. 258 tests green.
