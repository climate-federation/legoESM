# Round 221 — Decision 94: NEMO's enhanced vertical diffusion, repaired

**Decision 94 (user, operator note CF).** Round 220's plant found that the
seamount SMT-1 card states an enhanced vertical diffusion (NEMO `zdf_evd`)
and that multiplying its coefficient by ten thousand changed nothing. This
round repairs what that plant exposed. Two defects, both landed; neither is a
tuning choice.

**REVIEW.** One fresh adversarial reviewer on the diff: **SHIP WITH FIXES**
— "the NEMO transcription is right; what fails is the SCOPE of the fix".
Three MAJOR and three MINOR findings, all taken, each marked **[R-n taken]**
where it lands; two of them were real holes (a third call site, and a
wiring with no test). Codex is paused on this account, so DUAL review is a
stated GAP, not an exemption.

**Headline.** The trigger was reading the wrong fluid and the coefficient was
being added instead of replacing. Both are fixed on the shared path, every
card that runs the scheme now STATES which composition it runs (no default;
a card on NEMO's trigger raises when it is unset), and the eleven certified
cards plus the GYRE ladder are unmoved. Round 220's reading of its own plant
is **RETRACTED**: the coefficient did reach the solve — the trigger's 61
firing cells were all below the seafloor.

---

## 1. The trigger: NEMO tests its OWN rn2, built with the DECK's EOS

`zdfevd.f90` of the SMT-1 build's ppsrc
(`tests/VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/zdfevd.f90`):

```fortran
107  DO jk =  1,  jpkm1  ; DO jj = ... ; DO ji = ...
108     IF(  MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )      &
109        &  p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)
110  END DO   ;   END DO   ;   END DO
```

`rn2` is the model's own Brunt–Väisälä frequency from `bn2`, and `bn2` reads
the alpha/beta the deck's equation of state resolves (`eos.f90` `eos_rab` /
`bn2`; S-EOS on VORTEX, EOS-80 on ORCA2/GYRE, as each deck selects).

**The defect.** legoESM's `n2_mode="nemo_bn2"` trigger built alpha/beta from
`NemoSEOSConfig()`'s **DINO defaults** whatever fluid the card ran, because no
caller threaded the card's own `&nameos` coefficients — the comment above the
call said exactly that. On the seamount SMT-1 card (`rn_a0 = 0.28`, every
other coefficient zero, decision 69) that is a different fluid.

**The fix.** The card's `eos_nemo_seos` is threaded from the model config
through `make_ocean_physics` / `compute_vertical_K_profiles` to
`convective_K_A_flag`. `None` still means the defaults, which is what a card
stating no coefficients resolves to in its own density path — the card's
resolved value, not a substitute for it. **[R-6 taken]**: the field is
declared on the config, so it is read directly rather than through `getattr`,
which would silently keep the defaults on a rename.

**Measured, SMT-1 pristine initial state, fp64:**

| trigger coefficients | firing interfaces | min N² over wet interfaces |
|---|---|---|
| the card's S-EOS (`_VORTEX_SEOS`) | **0** | +9.000e-06 s⁻² |
| `NemoSEOSConfig()` defaults (the old code) | **61** | +7.243e-06 s⁻² |

and NEMO fires on none (round 220 §3). **The 61 are all BELOW the seafloor**
— 0 on wet interfaces, 61 on rock — which is the whole explanation of round
220's plant reading 0.0.

## 2. The composition: NEMO REPLACES, it does not add

`p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)` is INSIDE the `IF`, and
`zdfphy.f90:359` calls `zdf_evd` **after** the background/closure copy at
`zdfphy.f90:348-351` (`avt(ji,jj,jk) = avt_k(ji,jj,jk)`), so a fired
interface carries `rn_evd` and nothing else. The momentum arm is guarded by
`IF( nn_evdm == 1 )` (`zdfevd.f90:121`) and likewise **replaces** `avm`
(`:133-135`); with `nn_evdm = 0` `avm` is never touched. `ln_zdfddm = .false.`
on these decks, so `avs` follows `avt` — one shared coefficient, which is
what legoESM carries.

legoESM summed the scheme's coefficient onto the background and onto any
closure. `EnhancedDiffusionConfig.evd_composition` now states which
composition a card runs:

* **no default.** A card whose trigger is NEMO's own (`n2_mode="nemo_bn2"`)
  RAISES when it is unset; every other (Oceananigans-semantics) card keeps
  the additive composition it has always had. That scope is deliberate and
  stated, not an oversight.
* **`nn_evdm` is stated, not implied.** Under `"nemo_replace"`, `nu_conv`
  must be `0.0` (nn_evdm = 0) or exactly `K_conv` (nn_evdm = 1) — there is no
  NEMO configuration in between — `nu_bg` must be 0, and the trigger must be
  the hard switch.
* The fired set is read back as `evd >= convective`, which is **exact**: the
  stable branch is the strictly smaller `K_bg` and the coefficients are the
  config's own constants, never arithmetic.

Applied at every site that assembles the coefficient: the physics-provided-K
path in `_apply_implicit_vertical_mixing`, the fallback in
`compute_vertical_K_profiles`, and — **[R-1 taken]**, the finding that
mattered — the MPAS mesh path, which the first draft missed entirely.

## 3. Where the plant now moves T

On the SMT-1 card with one interior column made statically unstable **for the
card's own fluid** (T of the second wet level of column (31,31) warmed 2 K,
so the trigger selects exactly 1 interface):

| measurement | value |
|---|---|
| plant `rn_evd` 100 → 1e6, one step | **3.268e-02 K** |
| same plant on the PRISTINE card | **exactly 0.0** |
| `nemo_replace` − `additive`, unstable column | **2.735e-09 K** |
| `nemo_replace` − `additive`, pristine card | **exactly 0.0** |

The 2.735e-09 K is the `rn_avt0` background that used to ride on top of
`rn_evd`: NEMO's avt on that interface is 100 m²/s, legoESM's was 100.000012.

**RETRACTION.** Round 220 §7 read its plant as "the selection the card states
is not reaching the solve". That is wrong: it reaches the solve, and the
old (additive) composition moves T by the same 3.268e-02 K on this column.
The plant read 0.0 because the trigger's fired set was entirely rock, which
the wet-interface mask removes before the solve.

## 4. The regression tests, and what each fails on when reverted

`tests/ocean/fidelity/test_nemo_round221_evd.py` — 11 tests, 11 pass.

| test | fails when |
|---|---|
| trigger reads the card's EOS (0 vs 61) | `seos_cfg` is dropped, i.e. the defaults the old code always passed |
| the defaulted trigger fires only below the seafloor (0 wet / 61 rock) | the masking claim changes |
| **the replace wiring reaches the solve** (2.735e-09 K, and 0.0 pristine) | either integration hunk goes back to `+` **[R-2 taken]** |
| the explicit branch refuses the replacement | the construction-time guard is removed **[R-4 taken]** |
| the guards survive tracing (`jit(grad)` = 2.0) | a structural guard compares a traced leaf **[R-3 taken]** |
| composition is a replacement not a sum | the helper adds |
| a NEMO-trigger card must state its composition | a default is reintroduced |
| nn_evdm is stated not implied | the nn_evdm / nu_bg / smooth / K_bg guards are dropped |
| every NEMO card running EVD states NEMO's composition | a card stops stating it |

**[R-2 taken]** was the reviewer's sharpest point: the first draft's only
replace-vs-add assertion was a three-element array call on the helper, and
both model-stepping tests were `@pytest.mark.slow`, which the default suite
deselects — so reverting both integration hunks left every test green. The
wiring test above is not slow.

## 5. Which cards run EVD, and what each states

| card | trigger | `rn_evd` | `nn_evdm` | composition | deck line |
|---|---|---|---|---|---|
| `GYRE-zco` | `nemo_bn2` / teos10 | 100 | **1** (`nu_conv = K_conv`) | `nemo_replace` | `gyre_omip_l2_namelist_cfg` namzdf |
| `VORTEX_SMT1_VEC-zps` | `nemo_bn2` / seos | 100 | **0** (`nu_conv = 0`) | `nemo_replace` | ORCA2 rung-0 `namelist_cfg:409-411` |
| DINO `nemo_dino_kamm`, `…_mlf` (lat-lon) | `nemo_bn2` / seos | 100 | **1** | `nemo_replace` | DINO `namzdf` |
| DINO mesh (MPAS) builder | `nemo_bn2` | 100 | tracer-only | **`additive`, STATED** | the mesh path cannot express the replacement |
| every other card | `insitu` / `adiabatic` | — | — | `additive` (unchanged) | Oceananigans semantics |

The nine other certified NEMO cards carry no physics block at all, so they
are inert by construction — and measured below rather than assumed.

**A CHOICE MADE WITHOUT ASKING, flagged here and offered for revert:** the
DINO **mesh** (MPAS) card states `"additive"`. The MPAS convection path adds
a tendency, so NEMO's replacement cannot be expressed on it; the alternative
was to let that card raise. Stating the composition it actually runs changes
no number (it is what the mesh path has always done) and makes the deviation
visible, where raising would break a working card inside a repair round. The
lat-lon DINO cards are unaffected and state `nemo_replace`.

## 6. Gates

| gate | result |
|---|---|
| round-221 regression tests | 11 passed (9 default + 2 slow) |
| EVD / convection / recipe / DINO / MPAS / CLI modules | 209 passed |
| eleven certified card registries, 50 rows each | **0/50 rows moved each, TOTAL_MOVED_ROWS 0**; every first-over-bar unchanged (SMT-1, SMT flux, SMT vector, VORTEX±VEC at 30/15/10 km, LOCK_EXCHANGE, OVERFLOW) |
| GYRE certified ladder, 10 steps | 50 rows, **0 moved** vs round 217, first over bar kt=3 T/S/u/v/ssh — unchanged (re-run on the CLEAN committed tree 6f668d4222b9 after the reviewer's fixes; the first run carried a dirty stamp) |
| GYRE from-rest year, 360 days from rest, 8 scored days | **BIT-IDENTICAL to the certified pins at every scored day** (delta exactly 0.000e+00 K, 0.000 floor units, days 30/60/90/120/180/240/300/360); no re-pin, nothing to register under D59 |
| DINO from-rest month gate | **2.053801168e-03 K vs NEMO, against bar 2.244317642e-03 — PASS, unchanged to all ten printed digits** |
| receipt citation gate | 17 passed, `unmapped_citations == []` (51 map entries + 22 receipt lines re-anchored) |
| push battery (T4) | **136 passed** in 1096.92 s |

Three gates were ALREADY RED on the lane tip and are untouched by this round
(none of them reads a file this round edits):
`test_param_specs[shortwave_penetration.py]`,
`test_no_inline_physics_coeffs[land/restart.py]`,
`test_dispatch_hardening::test_no_dispatch_guard_removed`
(`land/slab_land.py::step_land`).

## 7. ORCA2 pointer

* **The two statements ORCA2 needs are both in.** Its rung-0 deck runs
  `ln_zdfevd = .true.`, `nn_evdm = 0`, `rn_evd = 100`
  (`namelist_cfg:409-411`) on a real ocean where convection fires — ORCA2
  round 140 measured rung-0 EVD active on **13,061 interfaces**. Its card
  takes the TKE closure, so it composes in `compute_vertical_K_profiles`:
  until this round that was `max(avt_tke, rn_evd)` under `nemo_max_floor`
  (and a straight sum under `additive`), where NEMO writes `rn_evd` alone.
  The two differ on every fired interface where the closure's own `avt`
  exceeds 100 m²/s, which in a convecting column is where EVD exists to act.
* **ORCA2's card must now STATE its composition**: its trigger is NEMO's, so
  leaving `evd_composition` unset raises. The value its deck pins is
  `"nemo_replace"` with `nu_conv = 0.0` (nn_evdm = 0).
* **How many interfaces fire with the card's EOS vs with the leaked
  defaults is NOT measured here** and is the ORCA2 lane's first item after
  the merge: ORCA2 selects EOS-80 (`ln_eos80`), so its `n2_eos_form` takes
  the Roquet polynomial branch, which never reads the S-EOS coefficient set
  — i.e. defect 1 is expected to be a NO-OP there and defect 2 is the whole
  of it. That expectation is a PREDICTION, not a measurement; the lane
  re-measures both numbers (fire count and the rung-0 ladder) after the
  merge, with the same plant used here.

## 8. OPEN

* DINO's month is unchanged to ten digits, which means either its EVD does
  not fire in the first month or the replace/add difference is below that
  precision. Not instrumented this round.
* GYRE is unmoved at ten steps AND over the whole year, to the last bit.
  Its deck runs `ln_zdfevd` with `nn_evdm = 1`, so either its closure never
  reaches an interface the trigger selects in a from-rest year, or it does
  and `avt_tke` there is already below `rn_evd` so replacing and max-flooring
  agree. Which of the two is NOT measured here; it is the same instrument
  ORCA2's lane will run, and GYRE is the cheaper place to run it.
* The reviewer could not check whether `zdf_phy` can be entered twice per
  step under RK3 tiling.
* The 100-day SMT-1 comparison, still preregistered and not run (round 220
  §11).
