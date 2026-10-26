# TSUNAMI round 8 receipt — the outer integrators read the config's y-wrap

Date 2026-10-09. Base `e1940fd76` (round 7). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round8/`.
Trigger: the operator's second codex re-review of the PR diff
(`0575754c0...HEAD`, packages + tests) found ONE remaining HIGH, CONFIRMED
finding; defaults unchanged, periodic copies match NEMO, 35 tests passed. No
NEMO statement is added or changed; this is a wiring fix, so there is no
preregistered measurement, only round 7's numbers as the bar.

## Disposition

| finding | disposition |
|---|---|
| HIGH: `_step_impl` is scoped to the config's y-wrap, but the outer AB2 / leapfrog / NEMO-MLF integrators build partial-cell face masks outside that scope (`ocean_model_latlon_cgrid.py:13781, 14444, 14931, 15224`), and `compute_face_masks_3d` reads the context flag (`latlon_cgrid_operators.py:3994`); a periodic partial-cell AB2 run under the default caller context walls its seam | **FIXED.** The four integrators (`_ab2_step`, `_leapfrog_step`, `_nemo_mlf_step`, `_unsplit_ab2_step`) join the wrapped list at `ocean_model_latlon_cgrid.py:15975-15978`, through the same `_honour_meridional_periodicity` wrapper as `_step_impl` (True and False). No new state, no new helper. Sites wrapped: 4 integrators. |

Why the integrators and not the four call sites: the sites sit inside
`_ab2_step` etc., which `_step_jitted` (`:13244`) calls BEFORE they call
`_step_impl`; the helpers they reach (`_apply_implicit_vertical_mixing`,
`_apply_tke_advection`, `_eke_3d_step`, `_tke_bottom_dirichlet`,
`_tke_step_entry_p_sh2`) are covered by the same wrap.

## Caller list (every `compute_face_masks_3d` caller; grep over `packages src scripts`)

| site | enclosing function | integrator path / scope |
|---|---|---|
| `ocean_model_latlon_cgrid.py:13781` | `_ab2_step` | AB2; NEWLY wrapped |
| `ocean_model_latlon_cgrid.py:14444` | `_leapfrog_step` | leapfrog; NEWLY wrapped |
| `ocean_model_latlon_cgrid.py:14931` | `_nemo_mlf_step` | NEMO MLF; NEWLY wrapped |
| `ocean_model_latlon_cgrid.py:15224` | `_unsplit_ab2_step` | unsplit AB2; NEWLY wrapped |
| `ocean_model_latlon_cgrid.py:1750` | `_nemo_dynzdf_drag_face_thickness` | called from `_apply_implicit_vertical_mixing` (`:12617`), reached from `_step_impl` and from the four integrators: scoped |
| `ocean_model_latlon_cgrid.py:5789, 6084, 7412, 8515` | inside `_step_impl` | scoped (round 7) |
| `ocean_model_latlon_cgrid.py:10682, 11043, 11225, 11431, 12339, 12376` | `_tke_bottom_dirichlet`, `_tke_step_entry_p_sh2`, `_apply_tke_advection`, `_eke_3d_step`, `_apply_implicit_vertical_mixing` | model-internal helpers reached only from the step paths: scoped |
| `ocean_pe_latlon_cgrid.py:3655, 5181`; `gm_redi_latlon_cgrid.py:267, 3409, 5919`; `mle_latlon_cgrid.py:313`; `k_profiles.py:791`; `barotropic_latlon_cgrid.py:321` | tendency / mixing / seed helpers | called from the model's step and tendency paths: scoped |
| `diagnostics_sections.py:753`, `diagnostics_streamfunction.py:89`, `fidelity/box_heat_budget.py:297`, `_future/immersed_boundary.py:157` | post-processing / future | outside the model; take no config; read the context flag (default False) |
| `scripts/validate/ocean_fidelity/**` (35 probe/gate sites) | probes and gates | outside the model; closed cards, `j_periodic` False |

## Gates (clean tree, commit `59cb0f413`)

| gate | result |
|---|---|
| plant A: integrators removed from the wrapped list | `2 failed` (the partial-cell AB2 topology test: periodic config under the default caller scope no longer equals the reference run under the True scope; and the plant test) |
| config-wiring tests with the fix | `26 passed` (card test file) |
| 100-step record, kt = 1..100 | `per_kt` 100/100 rows equal to round 7's final json; worst ssh **7.693e-16** normalised at kt 90, as round 7; json `0c73021deb868ade` (differs from round 7's file only in the git sha stamp, `59cb0f413b68`) |
| six closed cards, state sha256 after 3 steps, no caller scope | equal to round 7: LOCK_EXCHANGE `b3aa27fa80dbdefd`, OVERFLOW `72ad6c2d6a9be145`, GYRE `b8cb5ecd4dc4b467`, VORTEX `b0ef8aaf1dc75f91`, VORTEX_VEC `ce76bd63f0441d3f`, VORTEX_SMT4_VEC `1cc2f09098310c7f`; json `a8c2a1fc07d45b73`, byte-identical to round 7's |
| geometry gate | `GEOMETRY IDENTICAL` |
| TSUNAMI tests (card, check_records, geometry gate, ladder) + advection-off arms | `66 passed` + `11 passed` |
| push battery (TKE terms, recipe, real freshwater closure; ws stage face mask, prognostic barotropic state + gate, citation gate) | `109 passed` + `29 passed`; the file set is reconstructed from round 7's gate log plus two prognostic-state files, because round 7 recorded counts but not the exact list |

## Plant (the new test fails without the fix)

`tests/ocean/unit/test_nemo_tsunami_card.py`: a partial-cell (bathymetry
varying 1800-3200 m on 4 z* levels), meridionally periodic, `outer_integrator="ab2"`
8x16 lat-lon basin with eta bumps on the j seam.
- `test_partial_cell_ab2_outer_path_takes_its_topology_from_the_config`: the
  step under the default caller scope equals the step under a True caller scope
  and differs from the walled config.
- `test_partial_cell_ab2_plant_unscoped_outer_path_fires`: with the four
  integrators unwrapped (the plant) the periodic config under a False scope
  gives different bits.
- With the four names removed from the wrapped list both tests FAIL (run
  above, file restored from a copy afterwards; `git diff --stat` verified).

Not detectable: the opposite direction (walled config under a True caller
scope) gives the same AB2 bits with or without the wrap on this basin, so the
plant covers only the periodic-under-False direction the finding names.

## Boundary

Same as round 7: functions called outside the model (diagnostics, scripts,
the `_future` module) read the context flag, default False. Only the
integrator paths were in scope this round.

## Choices this round

| choice | ASKED or UNASKED |
|---|---|
| wrap the four integrators rather than each of the four mask sites | ASKED ("the same scoping helper around the four sites", sites live in these four functions) |
| plant basin: synthetic 8x16 partial-cell lat-lon, AB2 | UNASKED (test-only; the TSUNAMI card is full-cell) |

## Review

Operator reviews: round 6 BLOCK; round 7 fixes; re-review BLOCK on this one
site class; fixed here. Single review (codex) of the round-8 diff: see the PR
summary (Review section). NO GATE for dual review: GLM not run.

UNVERIFIED: the wrapper reads `kwargs["config"]` or `self.config`; the four
integrators are always called without a `config=` override from `_step_jitted`.

## Citations

The exchange the flag selects: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:1868`,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:2026-2033`.
