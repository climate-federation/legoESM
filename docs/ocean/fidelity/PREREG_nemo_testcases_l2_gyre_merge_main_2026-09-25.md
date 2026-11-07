# Preregistration — GYRE NEMO-fidelity lane merges GitHub `main`, 2026-09-25

Written and committed BEFORE `git merge github/main` runs.

## Scope

Lane tip `ce29fc3c9475` (branch `fidelity/nemo-testcases-l2-gyre-codex2`,
round 169 HELD) merges GitHub `main` `d3f624841`.  Merge base
`9f4b16d633f0adb52084b84646283238927baa12` — the merge commit of the
2026-09-17 merge-main round.  `main` has advanced 374 commits since.

No squash, no rebase: one ordinary merge commit with two parents.

## Expected textual conflicts (operator's dry run)

| file | why it is expected to conflict |
|---|---|
| `packages/core/legoesm/core/bulk_flux.py` | main's bulk-flux work vs the lane's NEMO bulk identities |
| `packages/core/legoesm/grids/tripole.py` | main's tripole work vs the lane's ORCA2 grid touches |
| `packages/ice/legoesm/ice/config.py` | main's SI3/sea-ice config vs the lane's SI3 card fields |
| `packages/ice/legoesm/ice/sea_ice.py` | same, in the model body |
| `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` | the shared lat-lon C-grid step — the GYRE NEMO path itself |
| `tests/test_validate_strict_coverage.py` | both sides add config fields that the strict-validation ratchet enumerates |

Conflicts outside this list, or an expected file that merges cleanly, are
recorded in the receipt as a deviation from this preregistration.

## Files edited on BOTH sides (26)

`git diff --name-only 9f4b16d633f0 github/main` ∩
`git diff --name-only 9f4b16d633f0 HEAD`:

```
packages/core/legoesm/constants.py
packages/core/legoesm/core/bulk_flux.py
packages/core/legoesm/core/precision.py
packages/core/legoesm/grids/latlon.py
packages/core/legoesm/grids/operators_latlon_cgrid.py
packages/core/legoesm/grids/tripole.py
packages/ice/legoesm/ice/config.py
packages/ice/legoesm/ice/sea_ice.py
packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py
packages/ocean/legoesm/ocean/eos.py
packages/ocean/legoesm/ocean/mpas_config.py
packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py
packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py
packages/ocean/legoesm/ocean/physics/shortwave_penetration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py
packages/ocean/legoesm/ocean/restart.py
scripts/cluster/omip_nemo/_ab_shear_d30.sbatch
scripts/run/run_omip_core2.py
tests/grids/test_tripole_internals.py
tests/ocean/unit/test_k_zeta_bih_resolution_scaling.py
tests/ocean/unit/test_ocean_model_fesom.py
tests/ocean/unit/test_tke_carried_coefficients.py
tests/test_validate_strict_coverage.py
```

Every file in this list that git auto-merges still gets its main-side hunks
read and given an execution verdict against the resolved GYRE NEMO card, as
in the 2026-09-16 and 2026-09-17 receipts.  The trajectory proof below is the
gate on that audit.

## Resolution rule, stated before any conflict is seen

1. Where a conflicting hunk carries a NEMO-transcribed statement, the side
   that matches the compiled NEMO source the lane cites wins, and the receipt
   quotes that `ppsrc` `file:line`.  Preference, tidiness and recency decide
   nothing.
2. Where main's change is orthogonal to the NEMO identities (sea ice, tripole,
   bulk flux for non-NEMO cards, MPAS), the resolution is the UNION: main's
   addition is kept verbatim and the lane's NEMO identities keep their exact
   routing.
3. The shortwave selector seam in `ocean_pe_latlon_cgrid.py` keeps the
   2026-09-17 precedence rule unchanged: `physics.shortwave_penetration.scheme`
   routes the NEMO identities, `surface_forcing.shortwave_scheme` routes every
   other value, and a card that sets both refuses with a named error.
4. `nemo_stage_momentum_wzv_split` stays a `bool | None` sentinel that each
   card states explicitly (GYRE `True`, ORCA2 `False`); unset still raises.
   No default moves.

## The claim this round will prove

**The merged tree's GYRE trajectory is bit-identical to the pre-merge lane
tip's.**  Concretely, measured on two arms run from clean committed trees
(before = this preregistration commit, after = the merge commit), with the
same instruments rounds 160–169 used:

| quantity | pass condition |
|---|---|
| certified trajectory ladder, 70 rows (`--compare-to`, `traj_comparison.json`) | `rows=70`, `max_worsening_ulps=0`, **0 rows moved**, and the two `ladder.json` documents equal |
| `ladder.residuals.npz` | every array `np.array_equal` |
| day-30 T rms | `6.5726e-05 K` on both arms, equal to the byte |
| day-240 T rms | `1.64484e-02 K` on both arms, equal to the byte |
| day-360 T rms | `1.12257e-02 K` on both arms, equal to the byte |
| day-by-day gap rows (360) | identical apart from the `worktree` provenance key |
| DINO / LOCK_EXCHANGE / OVERFLOW / generic-GYRE card gates | unchanged verdicts |

The run-to-run floor of this harness is ~2e-10 K (round 129); the claim here
is stronger than the floor — byte equality, not agreement within the floor.

**If any row moves**, the merge brought a shared-code change that reaches the
NEMO path.  The response is fixed in advance: find the `main` commit with
`git log -S` or a bisect over the merged files, cite it, and either route it
out of the NEMO identity through configuration (never by moving a default) or
HOLD the merge with the receipt naming the commit and the row.  Salvaging the
number by adjusting the instrument is out of bounds.

## Also preregistered

- The citation gate is re-anchored by RIGID shift only: a citation moves by a
  constant line delta and its extent is unchanged, after the cited text is
  verified textually identical between the lane parent and the merged tree.
  No citation is re-pointed at different text.  All nine self-test plants must
  fire.
- The full ocean-fidelity battery runs once, one battery at a time, and its
  failures are diffed against the lane's known-red list (5 held + the sea-ice
  red).  Anything new is a merge defect, not lane debt.
- Two independent reviews (a Claude code-reviewer subagent and `codex exec`),
  every finding closed or registered, the failing review re-run once.
