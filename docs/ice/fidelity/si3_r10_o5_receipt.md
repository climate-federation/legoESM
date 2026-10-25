# SI3 round 10 receipt — O5 model preparation: five-category H79 strength and aEVP on the ORCA2 tripole

## Own errors, first

1. My first comment and preregistration draft put NEMO's `at_i` recompute at
   line 962; it is `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:973`
   (`at_i(:,:) = SUM( a_i, dim=3 )`). Corrected before the preregistration commit.
2. My first synthetic fixture zeroed the metrics in the closed south halo, so
   the aEVP returned NaN and two plants "fired" only because of the NaN. The
   fixture now keeps the south-halo metrics finite, as a `domain_cfg` does, and
   the value-pin test asserts every output is finite.
3. My first card-level non-vacuity plant (parity swap in the lane-3 wrapper)
   was inert on ICE_ADV2D_RHG: that card's velocity updates do not couple
   through the parity order. Replaced by a perturbed stress-divergence weight,
   which turns the control red.
4. My first full-versus-partial fold plant was vacuous for a real reason (next
   section) and was rewritten to break the pivot-row symmetry.
5. My first citation-gate pass read only the gate's failure list, which omits
   unmapped citations; the gate's own status was FAIL. Every citation is now
   mapped and the status is read.
6. (Found by review.) The aEVP read the entry `u_ice`/`v_ice` in place of the
   preregistered `u_ice_b`/`v_ice_b`; its Coriolis lacked the no-FMA barriers;
   its "eager" side was a compiled loop; and this receipt misdescribed the
   ICE_ADV2D_RHG plant. All four fixed below.

## Outcome

O5's two exact-input pieces are in the one shared ice package, selectable, with
lane 3's cards bit-identical to the parent through their production
dispatchers. No NEMO array was read for O5; the acquisition is preregistered,
not built.

## Build cited

`ORCA2_SAS_ICE_O2_REF` (unpatched O2 reference build). Its `icedyn_rhg_evp.f90`,
`icedyn_rdgrft.f90`, `icedyn.f90` and `icevar.f90` are byte-identical (`cmp`)
to the other three ORCA2_SAS builds and (`diff`, empty) to lane 3's
`tests/ICE_ADV2D_RHG_OMIP_L3` build. So lane 3's aEVP transcribes the same
compiled routine; ORCA2 differs only in the halo program, five categories plus
smoothing in the strength, and the SSH operand.

## Search before build (RULE 4)

- H79: lane 3's `ice_strength(..., source_exact=True)` plus its guard inside the aEVP arm. Reused.
- aEVP: lane 3's `si3_cgrid_aevp_solver` (periodic halo, one category). Generalised, not copied.
- Strength smoothing: `git grep "str_smooth\|strength_smooth\|zworka"` at the parent: nothing. New.
- `ice_var_sshdyn`: `git grep sshdyn` at the parent: nothing. New.
- Fold: O1's `si3_lbc_lnk_orca2` had T/U/V only (its docstring already claimed F; the code raised). Extended, not re-implemented.

## What was transcribed

Strength, `ice_strength` H79 arm:
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:973`: `at_i(:,:) = SUM( a_i, dim=3 )` (sequential category sum).
- `:1081`: `strength(ji,jj) = rn_pstar * SUM( v_i(ji,jj,:) ) * EXP( -rn_crhg * ( 1._wp - at_i(ji,jj) ) )`, on the full halo extent, zero where `at_i <= epsi10`.
- `:1102-1105`: first line `zworka(ji,jj) = ( 4._wp * strength(ji,jj)`, last line `&            ) / ( 4._wp + tmask(ji-1,jj,1) + tmask(ji+1,jj,1) + tmask(ji,jj-1,1) + tmask(ji,jj+1,1) )`; interior only, reading the raw un-exchanged neighbours.
- `:1114`: `CALL lbc_lnk( 'icedyn_rdgrft', strength, 'T', 1.0_wp )`.
- Model: `packages/ice/legoesm/ice/dynamics.py:1898` (`def si3_h79_strength_raw(`, shared with lane 3) and `packages/ice/legoesm/ice/dynamics.py:1989` (`def si3_orca2_ice_strength(`).

aEVP, `ice_dyn_rhg_evp`:
- Strength enters at `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:268` (`CALL ice_strength`) and SSH at `:281` (`zsshdyn(:,:) = ice_var_sshdyn( ssh_m, snwice_mass, snwice_mass_b)`).
- Levitating arm `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1090`: `ice_var_sshdyn(:,:) = pssh(:,:)`; the embedded arm is refused.
- Exchanges, each with NEMO's nature, sign and `ldfull`: `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:238` (`fimask`, `'F'`, full: it runs at `kt == nit000`), `:440` (`zdelta`, `zp_delt`, `'T'`, no `ldfull`), `:647` and `:754` (`u_ice` `'U'` -1, `v_ice` `'V'` -1, `ldfull=.true.`), `:845` (`zs1`, `zs2` `'T'`, `zs12` `'F'`, no `ldfull`).
- Model: `packages/ice/legoesm/ice/dynamics.py:1255` (`def _si3_cgrid_aevp_core(`, the one body), `packages/ice/legoesm/ice/dynamics.py:1203` (`def si3_cgrid_aevp_solver(`, lane 3: periodic halo `packages/ice/legoesm/ice/dynamics.py:825`), `packages/ice/legoesm/ice/dynamics.py:2087` (`def si3_orca2_aevp_solver(`, ORCA2), `packages/ice/legoesm/ice/dynamics.py:2066` (`def si3_var_sshdyn(`).

Fold, F nature:
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbcnfd.f90:722` (`CASE ( 'F' )`, T-pivot), rows `ij2 = ipj - 2*ihls + jj - 2`; model `packages/ice/legoesm/ice/si3_orca2_state.py:1383`.
- When a call without `ldfull` folds the duplicated pivot line: always for one rank (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbclnk.f90:2112`, `IF( jpni == 1 )  THEN   ;   CALL lbc_nfd(`), else only at the first step (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbcnfd.f90:1617`). This is the explicit, default-free `nfd_full_when_absent` of `packages/ice/legoesm/ice/dynamics.py:1930` (`class SI3ORCA2FoldProgram(NamedTuple):`).

## Exact-input structure

(a) `si3_orca2_ice_strength(a_i, v_i, tmask, config, fold)`: NEMO's `ice_strength` entry arrays -> `at_i`, raw and smoothed strength.
(b) `si3_orca2_aevp_solver(state, forcing, metrics, config, strength_t, u_ice_b, v_ice_b, ssh_m, snwice_mass, snwice_mass_b, ln_ice_embd, fold, differentiable)`: NEMO's post-strength field and `icedyn_rhg_evp` entry arrays -> velocities and stresses. Every physical input is required; `None` raises; `forcing.ssh_t` must be `None` (the levitating arm owns SSH). `u_ice_b`/`v_ice_b` are the recorded previous-step velocities, never the entry state.

## Tests (`tests/ice/unit/test_si3_o5_strength_aevp.py`)

- F fold equals NEMO's index formulas written as independent 1-based loops (`tests/ice/unit/test_si3_o5_strength_aevp.py:172`).
- Strength equals an independent scalar C-`exp` transcription over every cell, coast included (`tests/ice/unit/test_si3_o5_strength_aevp.py:198`).
- Value pins (regression hashes, self-generated) plus plants: unmasked smoothing, exchange before smoothing, wrong fold sign on strength (`tests/ice/unit/test_si3_o5_strength_aevp.py:226`); wrong fold sign for each of `fimask zdelta zp_delt u_ice v_ice zs1 zs2 zs12` (`tests/ice/unit/test_si3_o5_strength_aevp.py:280`); parity swap. All fire.
- Nature, sign and `ldfull` at every exchange site, exact list (`tests/ice/unit/test_si3_o5_strength_aevp.py:291`).
- Full vs partial pivot fold (`tests/ice/unit/test_si3_o5_strength_aevp.py:317`): identical on fold-symmetric input (NEMO's "NP repro" parentheses make the mirrored stencil bit-identical), different once one pivot-row cell is asymmetric.
- jit == eager and central finite difference: strength (one `v_i` cell, ratio within 1e-6) and aEVP (strength scale, ratio within 1e-5), gradients finite and nonzero. The aEVP's eager side runs under `jax.disable_jit` with nonzero Coriolis shown to bind.
- Recorded `u_ice_b`/`v_ice_b` versus the entry-velocity substitution: both velocities differ (`tests/ice/unit/test_si3_o5_strength_aevp.py:400`).
- No defaults: missing or `None` physical inputs, a second SSH source, the embedded arm, a one-category config and an untyped fold program are all refused.
- Result after the review fixes: `20 passed, 1 deselected in 102.40s` (evidence `resume/o5_unit.log`).

## Lane-3 parent-vs-current control (production dispatchers, parent `1bb17df9dd`)

- ICE_ADV2D_RHG, `step_ice_adv2d_rhg_card`, two steps, every state leaf equal. The committed test also plants stress-divergence weight 0.5001 in the PARENT wrapper and asserts the first-step `u_ice` then differs. After the review fixes: passed inside `resume/o5_unit.log`. (The earlier separate plant run, `parent_control_rhg_PLANT.log`, failed with 24,240 of 169,744 elements different; that log does not record which weight was planted, so it is not cited for one.)
- ICE_RHEO, `step_ice_rheo_card`, one full step, after the review fixes: `1 passed, 20 deselected in 489.30s` (`resume/parent_control_rheo.log`). Marked slow.
- Both lane-3 cards run with zero Coriolis, so the new Coriolis rounding cannot reach them; the other three aEVP consumer suites: `22 passed in 288.33s` (`resume/aevp_consumers.log`).
- O1/O2/O3 suites after the fold change: `37 passed in 211.33s` (`o123.log`, `c0179c4ca517f938`).

Evidence directory: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round10/`.

## Review fixes (resume)

- `u_ice_b`/`v_ice_b` (codex): required inputs, `icedyn_rhg_evp.f90:574,625`. Plant (substitution restored in the solver): the new test fails (`resume/item1_PLANT_substitution.log`); reverted, passes.
- Coriolis no-FMA (Argon): every operation of `icedyn_rhg_evp.f90:560-562,611-613,666-668,718-720` rounded in source order. Without it the real eager-vs-jit test fails, 5 of 192 `u_ice` cells (`resume/item3_before_barriers.log`); with it, passes.
- Eager-vs-jit (Argon): eager now means `jax.disable_jit`.
- Lane 3 keeps passing its entry velocity as `u_ice_b`: `store_fields` sets `u_ice_b = u_ice` before `ice_dyn` (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90:463`). Unchanged behaviour, offered for review.

## Gates

- Citation gate, all eight SI3 receipts including this one: status PASS, 0 unmapped, 0 map entries failing audit (`citation_gate_all_si3.log`, `f212130e230e563f`); a planted citation shift turns this receipt red. O2/O3 receipt citations into `si3_orca2_state.py` moved +17 lines and were re-anchored.
- Default citation gate: `17 passed in 10.73s` (`default_gate.log`, `aa9dd98312bfef62`).
- Private-import, inline-coefficient (`dynamics.py`) and hardcoded-constant ratchets on the touched files: `25 passed` (`ratchets.log`, `ed786ef0557ac0d4`). The 30 reds of the wider batch (`existing_b.log`) are pre-existing sites outside this diff.
- Review: codex review: operator. One in-round Claude review: no confirmed defects; its three test-gap notes were closed (whole-interior reference, site-by-site exchange check, test rename).
- GPU spot check: not run (no certification this round).

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: lane 3 bit-identical (both cards); strength equals an independent transcription; F fold equals NEMO's formulas; all plants fire; jit == eager; FD agrees.
- PLAUSIBLE: O5 bit-exactness against NEMO (no record yet; preregistered P1-P7); `at_i` at aEVP entry equals NEMO's recomputed `at_i` (P2); the multi-rank partial fold (one-rank decks never run it).

## CHOICES

- Build cited = `ORCA2_SAS_ICE_O2_REF` (byte-identical alternatives). UNASKED, mechanical.
- `nfd_full_when_absent` exposed as a required input rather than inferred. UNASKED; offered for revert.
- Acquisition on one MPI rank, following O1/O2. UNASKED; offered for revert (preregistration section 1).
- Inner frames `jter in {1, 2, 3, 50, 99, 100}` (design: "selected 100-subcycle frames"). UNASKED; offered for revert.
- Step window `nn_itend = 2`. ASKED (DECISION_NEEDED).
- Test-local subcycle count 4 and synthetic 16 x 12 grid. Tests only.
- No configuration default or production config changed.
