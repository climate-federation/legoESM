# Decision 35 preregistration — the GYRE card drops `fix_eta_drift`

Date: 2026-09-11. Frozen base: `3a8d94ea1985` (round 54). CPU, production
JIT, fp64/scalar-libm only. NEMO source and records are read-only. Written
BEFORE the after-arm is measured; the before-arm runs at the frozen base in a
detached worktree.

## The decision (ASKED, user 2026-09-11)

Question put: the certified GYRE NEMO-identity card applies a global sea-level
correction every step (`fix_eta_drift=True`, a uniform eta shift sized by an
area-weighted volume residual, `ocean_model_latlon_cgrid.py:6686-6751`) that
NEMO does not have. Options: (1) turn it off on the GYRE card; (2) leave it on.
User: **(1)** — "no hidden extras; seems like a fixer that could hide model
errors."

## What NEMO does (compiled branch `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo`)

- `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90:137`: `pssh(ji,jj,Kaa) = ( pssh(ji,jj,Kbb) - rDt * ( r1_rho0 *
  emp(ji,jj) + zhdiv(ji,jj) ) ) * ssmask(ji,jj)` — emp enters LOCALLY.
- `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/usrdef_sbc.f90:151-157`: NEMO de-means the card's own emp
  (`emp(ji,jj) = emp(ji,jj) - zsumemp * tmask(ji,jj,1)`), so the area-mean
  source is zero by construction (measured `5.5e-22` kg/m2/s, card
  reconciliation receipt §9.1).
- No global sum exists anywhere in NEMO's free-surface path.

## What changes

1. `nemo_testcase_recipe.py` (GYRE branch only): `fix_eta_drift=True -> False`;
   `freshwater_closure="real_freshwater"` stays (the `none` arm KILLS the
   channel: `3.95e-03` K after two steps, reconciliation receipt §7).
2. `ocean_model_latlon_cgrid.py`: the #1484 guard that forces
   `fix_eta_drift=True` under `real_freshwater` is exempted ONLY when
   `barotropic_continuity_evaluation="nemo_literal"`, i.e. when the source
   enters eta by NEMO's own substep statement. The generic lane keeps the
   guard (its budget claim was measured on that lane). A guard relaxation
   admits a new combination and changes no arithmetic of any configuration
   that constructs today.
3. `nemo_testcase_l2_gyre_phase3_gate.py`: the gate's `_replace(...,
   fix_eta_drift=True)` override is removed so the ladder certifies the
   card's own program (one card, one program).

## Predictions (falsifiers stated)

Ladder, kt=1..10, scalar-math v2 roots, before vs after:

- `kt2.before.ssh`: `4.336808689942018e-19` -> **`0.0`, 0 cells unequal**. The
  reconciliation receipt attributed the entire kt=2 difference to this field;
  the fixer's shift is the whole residual. FALSIFIER: any nonzero kt2 ssh row
  → the residual has a second owner.
- `kt2.before.u/v`: **unchanged bit-for-bit** (`2.7478404751243857e-12` /
  `3.305560306813421e-12`); the fixer acts on eta after the momentum update.
  FALSIFIER: either row changes at all.
- `kt2.before.T/S`: **unchanged** (`6.054e-16` / `5.786e-16` normalised).
  FALSIFIER: either row changes.
- kt>=3 rows: expected to move within their existing DEBT (`8.74e-3` T at kt3
  is the step-2 tracer owner, round 54); every moved row is registered.
  First-over-bar stays `kt2 u/v`. FALSIFIER: a first-over-bar earlier than
  kt2, or any AT-BAR row leaving the bar.

Days 1–30 (year_owners `--day-gap`, NEMO `nemo_seed0` daily restarts):

- the 3-D T rms gap at every day equals the before-arm to >= 4 significant
  digits (the ON-OFF seed is `1.9e-10` K at step 12 against a `1.42e-02` K
  gap at day 30). FALSIFIER: any day's gap moving by more than `1e-4`
  relative.

Rule-12 per card: GYRE measured (above). LOCK_EXCHANGE / OVERFLOW: the L1
base resolves `fix_eta_drift=False` with the library freshwater closure and
no surface freshwater — they never construct the admitted pair and execute
no changed statement (resolved pairs printed in the receipt). DINO: separate
branch; shared-statement risk = the guard only (a relaxation; 0 rows by
construction). ORCA2: UNMEASURED-with-spec.
