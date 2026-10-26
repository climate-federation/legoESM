# NEMO testcase L2 GYRE — the TKE runaway: alignment, preregistration, receipt

Date: 2026-09-11.  CPU / fp64 / `PrecisionPolicy.fp64(transcendentals="libm")`.
Worktree `/tmp/codex-gyre`, branch `fidelity/nemo-testcases-l2-gyre-codex2`,
preregistered at `d8e97e1326de` (clean).  Oracle read-only at
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`; every NEMO citation below is a
line of the GYRE card's OWN compiled source,
`cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90` (cited `zdftke:NNN`),
resolved by `cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/EXP00/namelist_cfg` +
`cfgs/SHARED/namelist_ref`.

## The card, as the namelists resolve it (Rule 10 — instantiated and printed)

| NEMO namelist | value | source | legoESM `TKEConfig` field (printed from the built card) |
|---|---|---|---|
| `ln_zdftke` | `.true.` | cfg `namzdf` | `vertical_mixing.scheme="tke"` |
| `rn_Dt` | `14400.` | cfg `namdom` | `card.dt_s = 14400.0` |
| `jpkglo` | `31` | cfg `namusr_def` | `z_coord.n_levels = 30` (T-cells); TKE rows `N = 29` |
| `rn_ediff` | `0.1` | ref | `c_k = 0.1` |
| `rn_ediss` | `0.7` | ref | `c_eps = 0.7` |
| `rn_ebb` | `67.83` | ref | surface Dirichlet (`surface_bc="nemo_dirichlet"`) |
| `rn_emin` | `1.e-6` | ref | `tke_background = 1e-06` |
| `rn_emin0` | `1.e-4` | ref | `tke_surface_min = 0.0001` |
| `rn_bshear` | `1.e-20` | ref | `bshear_floor = 1e-20` |
| `nn_pdl` | `1` | ref | `prandtl_mode = "nemo_ri"` |
| `nn_mxl` | `3` | ref | `tke_mxl_choice = 3` |
| `ln_mxl0` / `rn_mxl0` | `.true.` / `0.04` | ref | `mxl0_min_m = 0.04` |
| `ln_lc` / `rn_lc` | `.true.` / `0.15` | ref | `lc = True`, `lc_coeff = 0.15` |
| `nn_etau` | `0` | **cfg** (overrides ref `1`) | `etau_mode = "none"` |
| `ln_drg_OFF` | `.false.` (ref default; cfg sets `ln_non_lin=.true.`) | ref `namdrg` | `bottom_tke_bc = False` — see row **N7** |
| `nn_eice` | `1` | ref | `eice = 0` (no ice model on this card; `fr_i` absent) |
| `cpl_phioc`/`ln_phioc` | not coupled | — | wave surface-BC arms inert both sides |
| `ln_zdfevd` / `nn_evdm` / `rn_evd` | `.true.` / `1` / `100.` | **cfg** `namzdf` (ref default `.false.`) | `convection.scheme="enhanced_diffusion"`, `K_conv=nu_conv=100.0`, `n2_threshold=-1e-12`, `two_level_trigger=True` — LIVE, added after review A flagged it missing from this table |

Geometry, printed: `bottom_level` is `29` on every wet column and `-1` on land
(`np.unique` over the 22x32 field); `is_active` sums to `30` on every wet column.
So NEMO's `mbkt = 30` (1-based) everywhere wet — a FLAT bottom — and
`mbkt+1 = 31 = jpk`.

**Index map, established here and used by every row below.**  legoESM carries the
`n_levels-1 = 29` INTERIOR W-interfaces, index `k = 0..28`; the z=0 surface W-point
is a separate virtual row prepended by `tke_surface_bc_level="nemo_z0"`
(`tke.py:1621-1633`).  The assembled system therefore has `30` rows
representing NEMO's `jk = 1..30`:

```
extended row 0   <-> NEMO jk = 1     (z=0 surface, Dirichlet)
extended row i   <-> NEMO jk = i+1
extended row 29  <-> NEMO jk = 30 = jpkm1      <-- legoESM interior index 28
(no legoESM row) <-> NEMO jk = 31 = jpk
```

NEMO's `jpk` row is legitimately absent: `en(jpk)` is read by NOTHING.  The
back-substitution seeds at `jpkm1` WITHOUT the `zd_up(jpkm1)*en(jpk)` term
(`zdftke:468`), and `tke_avn` loops `jk = 1, jpkm1` (`zdftke:681-687`), so
`en(jpk)` never reaches `avm`/`avt`/`dissl`.

## N1..N9 — zdftke dissipation and bottom boundary, statement by statement

| # | NEMO statement (GYRE-compiled) | legoESM | verdict |
|---|---|---|---|
| N1 | `zfact2 = 1.5*rn_Dt*rn_ediss` (`zdftke:246`), `zfact3 = 0.5*rn_ediss` (`zdftke:247`) — note `zfact3` carries NO `rn_Dt`; the `rn_Dt` arrives from the RHS factor at `:422` | `1.5*dt*cfg.c_eps` (`tke.py:1454-1455`) and `0.5*diss_rate` inside a `dt*(...)` (`tke.py:1484-1486`) | MATCH |
| N2 | `zdiag(jk) = 1 - zzd_lw - zzd_up + zfact2*dissl(jk)*wmask(jk)` (`zdftke:419`), for `jk = 2..jpkm1` — the IMPLICIT half, at the SAVEd preceding-step `dissl` | `literal_diag` (`tke.py:1453-1456`) — but built only for rows `0..N-2`, i.e. NEMO `jk = 2..29`. **NEMO `jk = 30 = jpkm1` is not built.** | **DIFF (N2b)** |
| N2b | — | `diag = concat([literal_diag, ones_like(...)])` (`tke.py:1457-1459`) hard-sets the deepest carried row's diagonal to `1.0` | **DIFF — the implicit half is deleted at `jk = jpkm1`** |
| N3 | `en(jk) = en(jk) + rn_Dt*( p_sh2 - p_avt*rn2 + zfact3*dissl*en(jk) )*wmask(jk)` (`zdftke:422-425`) — the EXPLICIT half, a POSITIVE source | `rhs = rhs_base + dt*(P_s + buoy_source + 0.5*diss_rate*rhs_base)*w_active` (`tke.py:1484-1486`), built for ALL `N` rows | MATCH in form; but at the deepest row it is the ONLY surviving dissipation term (see N2b/N5) |
| N4 | `zd_up(jk)=zzd_up`, `zd_lw(jk)=zzd_lw` (`zdftke:417-418`), `zzd_up` uses `p_avm(jk+1)+p_avm(jk)` and `e3t(jk)*e3w(jk)` (`zdftke:412-413`); `zzd_lw` uses `p_avm(jk)+p_avm(jk-1)` and `e3t(jk-1)*e3w(jk)` (`zdftke:414-415`); `zcof = zfact1*tmask(jk)` (`zdftke:409`) | `literal_up` / `literal_lw` (`tke.py:1368-1371`), rows `0..N-2` only; `a_diff` gets a TRAILING ZERO and `c_diff` a trailing zero (`tke.py:1372-1378`) | **DIFF — `zzd_lw(jpkm1)` is deleted** (the trailing zero in `a_diff`). `c_diff`'s trailing zero is CORRECT (`:468` drops the `jk+1` coupling), but `zzd_up(jpkm1)` must still sit in `zdiag(jpkm1)` and does not |
| N5 | Solve is `jk = 2..jpkm1`; `en(jpkm1) = zd_lw(jpkm1)/zdiag(jpkm1)` (`zdftke:468`); reverse pass `jk = jpk-2 .. 2` (`zdftke:470-472`) | `_nemo_literal_tke_solve` sets `jpkm1 = a.shape[-1] - 2` (`tke.py:965`, comment `:963` "For an extended length jpk, Python index jpk-2 is Fortran jpkm1").  The extended array has length **30 = jpk-1**, not `jpk`. So `jpkm1` evaluates to `28` (NEMO `jk = 29`) and the deepest carried row (NEMO `jk = 30 = jpkm1`) is EXCLUDED from both scans and returned as its RAW right-hand side (`tke.py:1021-1023`) | **DIFF — THE OWNING STATEMENT** |
| N6 | `p_avm(jpk)`: `tke_avn` writes `jk = 1..jpkm1` only (`zdftke:681-687`); `zdfphy.f90:226-228` initialises `avm_k(:,:,jk) = avmb(jk)*wmask(:,:,jk)` and `wmask(:,:,jpk)=0`, so `p_avm(:,:,jpk) = 0` for the whole run | legoESM carries `K_M_old` with `N = 29` rows (`jk = 2..30`) and `K_M_surface` (`jk = 1`); no `jk = 31` slot | MATCH by construction once `zzd_up(jpkm1)` is formed with a ZERO upper neighbour |
| N7 | Bottom BC `en(mbkt+1) = MAX(zebot, rn_emin)*ssmask` (`zdftke:284-292`), executed because `ln_drg_OFF=.false.` | `bottom_tke_bc = False` on this card — no bottom Dirichlet row | **INERT ON THIS CARD, not waived**: `mbkt+1 = 31 = jpk` on every wet GYRE column, and `en(jpk)` is read by nothing (`zdftke:468`, `:682`). The value NEMO computes is dead on GYRE. Registered as DEBT for any card with `mbkt < jpkm1` (DINO, ORCA2), where the row IS live |
| N8 | Post-solve floor `en(jk) = MAX(en(jk), rn_emin)*wmask(jk)`, `jk = 2..jpkm1` (`zdftke:473-475`) | `jnp.maximum(solved, floor) * w_active` (`tke.py:1023-1025`) | MATCH |
| N9 | Mixing-length bottom seed: `zmxlm(:,:) = rmxl_min`, `zmxld(:,:) = rmxl_min` (`zdftke:593-594`) before the `nn_mxl=3` ladders, so the `ldown` recursion at `jk = jpkm1` reads `zmxlm(jpk) = rmxl_min` (`zdftke:665-668`) | `compute_mixing_lengths` (`tke.py:664`) — bottom seed of the up-ladder | NOT RE-CERTIFIED THIS ROUND; carried as UNMEASURED. It changes `dissl` (hence the runaway's rate constant) but cannot change its SIGN, so it cannot own the blow-up |

## PREREGISTRATION (frozen before any measurement of the corner column)

**P1 — the owning statement.**  ONE transcription spread over three statements,
which must move TOGETHER.  Review A proved (and I accept) that the PROXIMATE
owner is **`tke.py:1457-1459`, the trailing identity diagonal**, not
`tke.py:965`: with `a[-1]=0` and `b[-1]=1` the solver's `terminal =
work/diagonal` equals the raw right-hand side for EITHER value of `jpkm1`, so
changing `:965` alone changes nothing at the runaway cell and would merely
push the bad value up into NEMO `jk=29` through `c_ext[28]`.  P1 is re-headed
accordingly; the original headline is RETRACTED.  The three statements are
`tke.py:1457-1459` (trailing identity diagonal), `tke.py:1372-1375` (trailing
zero sub-diagonal, and the N-1-row `literal_up`/`literal_lw`), and
`tke.py:965` (`jpkm1 = a.shape[-1] - 2`).  legoESM's TKE array carries
NEMO's W rows `jk = 1..jpkm1` and has NO `jk = jpk` row, so the last row must be
SOLVED, not held.  Held, it returns its raw right-hand side, which contains
NEMO's explicit dissipation add-back `+rn_Dt*0.5*rn_ediss*dissl*en`
(`zdftke:422-425`) with no implicit `+1.5*rn_Dt*rn_ediss*dissl` on the diagonal
(`zdftke:419`) and no diffusion coupling to the row above.  With
`dissl = sqrt(e)/L` that row's map is exactly

```
e(n+1) = e(n) + (0.5*rn_ediss*rn_Dt/L) * e(n)^1.5
```

which is the previous round's measured `16.4437` at `L = 306.50 m`, at the cell
the previous round's census pinned: `j=1, i=30, k=28` — and `k = 28` IS the
deepest carried row, `N-1`.

**P2 — the falsifier, before the fix.**  Measured at `(j=1, i=30, k=28)`, steps
40..48: the row's three assembled coefficients must be exactly
`a = 0`, `diag = 1`, `c = 0`, and `e(n+1)` must equal that row's RHS to
roundoff, with the RHS's dissipation term supplying at least 99 % of the
increment `e(n+1) - e(n)`.  If the increment is NOT dominated by
`dt*0.5*c_eps*dissl*e`, P1 is **REFUTED** and the owner is elsewhere.

**P3 — the falsifier, after the fix.**  With N2b/N4/N5 transcribed:
(a) the corner cell's `e(n+1)/e(n)^1.5` must stop being `~16.44`;
(b) the domain maximum TKE must stay below `1 m^2/s^2` through step 200 (NEMO's
    GYRE TKE is `O(1e-6..1e-2)`; `1` is four orders of slack, a bound not a
    tuning);
(c) the card must complete the 2160-step year with every watched field finite.
If (c) fails, the round is **REFUTED as a cure** — the fix stays (Rule 12) and
the next owner is named from the census at the new failure step.

**P4 — the kt=1..10 ladder.  RETRACTED BEFORE MEASUREMENT, by review A.**
My frozen prediction was "NOT bit-identical", on the grounds that the deepest
row's sub-diagonal `zzd_lw(jpkm1)` is non-zero from the first step.  Review A
refuted the *mechanism* with arithmetic I re-derived and accept: at `kt = 1`
from rest the deepest row's right-hand side is
`e + dt*(-avt*N2 + 0.5*rn_ediss*dissl*e)`, whose stratification sink
(`-1.7e-7`) is an order of magnitude larger than the dissipation add
(`+1.6e-8`), giving `rhs = 8.4e-7 < rn_emin = 1e-6`.  BOTH arms are then
clamped to exactly `1e-6` by the post-solve floor (`zdftke:473-475`), so the
deepest interface does not move at `kt=1` and a BIT-IDENTICAL ladder would NOT
refute the claim.  Review A also corrected my `zzd_lw` arithmetic: `e3t*e3w` is
`9.0e4`, not `2.5e5`, so the coupling is `~1.6e-6`, not `~7e-6`.
Replacement prediction, frozen here: **bit-identical at `kt=1`; whether the
floor still hides the row at `kt=2..10` is UNPREDICTED and is measured.**
If the ladder moves and WORSENS, that is a Rule-12 exposure: kept, registered
as debt, never reverted.

**P5 — other cards.**  DINO and ORCA2 run the same `_nemo_literal_tke_solve`
and the same literal matrix, with the same `n_levels-1` interface count, so the
same off-by-one is present there.  They differ from GYRE in having
`mbkt < jpkm1` columns, where the row NEMO holds (`mbkt+1`) IS inside the
carried range and legoESM's `w_active` already makes it an identity row.
Predicted: DINO's `kt = 1` closure fields change ONLY on full-depth columns
(`mbkt = jpkm1`).  The tanks (LOCK_EXCHANGE, OVERFLOW) are predicted NOT to run
the TKE closure at all — to be confirmed by citation, not assumed.

## MEASUREMENT — before the fix (model code at `d8e97e1326de`)

The model source was the preregistered commit; the only working-tree edit at
this point was the `--census-corner` option added to the already-committed
census probe, which reads carried state and computes nothing the model uses.
The `tke.py` transcription was applied only AFTER this log closed.  Evidence
`tke_runaway/corner_before.log`.

P2's operative clause was the 99 %-dominance one (review A is right that
`a=0, diag=1, c=0` is guaranteed by construction and so is source-reading, not
measurement).  It fired:

| kt | e(n) | e(n+1) | dissl(n-1) | `dt*0.5*c_eps*dissl*e` | actual increment | explicit share | (e(n+1)-e(n))/e(n)^1.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 1.79561e-03 | 3.03808e-03 | 1.38037e-04 | 1.24922e-03 | 1.24247e-03 | 1.005433 | 16.329257 |
| 38 | 1.29827e-02 | 3.72610e-02 | 3.71264e-04 | 2.42928e-02 | 2.42783e-02 | 1.000597 | 16.412423 |
| 41 | 1.16146e+00 | 2.17302e+01 | 3.51377e-03 | 2.05687e+01 | 2.05687e+01 | 1.000002 | 16.432408 |
| 44 | 1.14091e+06 | 2.00402e+10 | 3.48496e+00 | 2.00391e+10 | 2.00391e+10 | 1.000000 | 16.443864 |
| 47 | 1.65688e+26 | 3.50687e+40 | 4.19950e+10 | 3.50687e+40 | 3.50687e+40 | 1.000000 | 16.443039 |

Explicit share `1.000002` at step 41 and `1.000000` from step 42 on, against
the `>= 0.99` falsifier.  The ratio converges to `16.443869`; the prediction
from the census's own MEASURED `L` at that step (`306.4972 m`, not the rounded
`306.50` the previous round quoted) is `16.443870`, i.e. the agreement is one
part in `1e7`, not the one part in `1e4` implied by quoting `16.4437`.  The
neighbour row `k-1` sat at exactly `1.0e-06` at EVERY one of those steps — the
row is decoupled in both directions, which is the `a=0`/`c=0` structure showing
up in the data rather than in the source.  `u`, `v`, `T`, `S` went non-finite
at step 47, which is where the census catches it; the model's own abort is one
step later, at 48.

**Operator-level confirmation, independent of the model.** With a 30-row system
(NEMO `jk=1..30`) the solver returned its deepest row as exactly its own
right-hand side (`30.0` against `rhs[-1] = 30.0`) while solving the row above
(`15.2757`).  After the transcription it returns `15.8034`, and the whole
returned vector matches an independent dense solve of NEMO's own system —
built from `zdftke:419` for the diagonal and `zdftke:468` for the missing
`jk+1` coupling in the last row — to `4.44e-16`.

## MEASUREMENT — after the fix

`tke_runaway/corner_after.log`, 250 steps of the same card.

P3(a) **HELD**.  The corner cell's step ratio `(e(n+1)-e(n))/e(n)^1.5` is
`0.000000` at every printed step from 79 on; the cell sits on `rn_emin` and the
domain's TKE maximum has moved to the SURFACE (`k=0`), where a wind-driven
closure should put it.

P3(b) **HELD**.  The run's LARGEST domain TKE over all 250 steps is
`2.852e-03 m^2/s^2` at `kt=10` (`corner_after.log`, the `kt=  10` row) — NOT the
`1.979e-03` at `kt=240` that an earlier draft of this receipt and the commit
message quoted, which is that one step's value.  Either way it is against a
`1 m^2/s^2` bound, with two and a half orders of slack.

THE CONTROLLED COMPARISON, at a MATCHED step.  A previous draft said the
dynamical fields "stay in the ranges the healthy first 46 steps had"; that
compared a 250-step window against a 46-step one and is withdrawn.  At the
SAME step, `kt=40`, before and after:

| field at kt=40 | before | after |
|---|---|---|
| eta | `[-4.534e-02, 4.394e-02]` | `[-4.534e-02, 4.394e-02]` |
| u | `[-8.022e-02, 5.500e-02]` | `[-8.022e-02, 5.500e-02]` |
| v | `[-3.327e-02, 1.493e-01]` | `[-3.327e-02, 1.493e-01]` |
| T | `[0, 2.380e+01]` | `[0, 2.380e+01]` |
| max TKE | `1.161e+00` at `(j=1,i=30,k=28)` | `1.986e-03` at `(j=1,i=10,k=0)` |
| max avm | `3.241e+01` | `1.336e-01` |

Every dynamical row agrees to printed precision at step 40 and only the closure
differs — which is the point, and is a stronger statement than the withdrawn
one.  P3(c) is the year, below.

## MEASUREMENT — the kt=1..10 ladder, legoESM against itself (P4)

`--ladder-dump` on the same committed harness, at `d8e97e1326de` and at this
commit, 130 arrays, compared with `np.array_equal`
(`tke_runaway/ladder_compare.log`):

| kt | fields differing / 13 | max abs diff over all fields |
|---:|---:|---:|
| 1 | **0** | **0.0** |
| 2 | 8 | 4.7520e-04 |
| 3 | 12 | 1.0601e-03 |
| 5 | 12 | 2.6663e-03 |
| 10 | 12 | 1.1285e-02 |

**kt=1 is BIT-IDENTICAL**, exactly as review A's replacement prediction said it
would be: the deepest row's right-hand side is below `rn_emin` at `kt=1` and the
post-solve floor (`zdftke:473-475`) clamps both arms to the same `1e-6`.  From
`kt=2` the ladder moves; at `kt=10` the largest change is in the eddy viscosity
(`1.13e-02 m^2/s`, on the deepest interface), and the velocities move by
`1.4e-07 m/s`.

**Whether that move is TOWARD NEMO is UNMEASURED**, and the against-NEMO gate
disagrees with this self-A/B at kt=2 — see "THE AGAINST-NEMO LADDER" below, which
records both and names the configuration difference.  Nothing in this round is
bit-exact against NEMO.  "FIXED" in the Rule-12 table below means the runaway is
gone and the card completes, NOT that the card is at the exact bar.

## Retractions

None yet this round.  The previous round's surviving claim — that the runaway is
the explicit half of the dissipation split acting with nothing implicit against
it — is CONFIRMED and is now given its statement: it is not that NEMO's split is
mis-transcribed (it is not; N1 and N3 MATCH), it is that one ROW is excluded
from the solve.

## THE TRANSCRIPTION THAT LANDED

Three statements, one change, in the ONE shared TKE implementation
(`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py`).  No clip, no
floor, no limiter, no damping was added; nothing was made card-specific.

| # | before | after | NEMO |
|---|---|---|---|
| 1 | `diag = concat([literal_diag(N-1 rows), ones_like(1)])` | `diag` is `1 - zzd_lw - zzd_up + zfact2*dissl*wmask` on ALL `N` rows | `zdftke:419`, `DO jk = 2, jpkm1` at `:407` |
| 2 | `literal_up`/`literal_lw` over `N-1` rows; `a_diff` ends in a zero | both over `N` rows; `a_diff` ends in `zzd_lw(jpkm1)`.  `c_diff` still ends in zero — `zdftke:468` seeds `en(jpkm1)` WITHOUT the `zd_up(jpkm1)*en(jpk)` term — while `-zzd_up(jpkm1)` stays in the diagonal | `zdftke:412-418`, `:468` |
| 3 | `jpkm1 = a.shape[-1] - 2` | `jpkm1 = a.shape[-1] - 1` | the array is `jk = 1..jpkm1`; `en(jpk)` is read by nothing (`zdftke:468`, `:681-687`) |

`zzd_up` at the deepest row needs `p_avm(jpk)`, which NEMO never writes:
`zdfphy.f90:226-228` sets `avm_k(:,:,jk) = avmb(jk)*wmask(:,:,jk)` with
`wmask(:,:,jpk) = 0`, and `tke_avn` loops `jk = 1, jpkm1`.  The transcription
therefore pads the upper neighbour with an explicit ZERO rather than
re-using the row's own viscosity.

Also landed, from review B: the literal matrix now REFUSES
`tke_buoyancy_sink != "nemo_explicit"`.  `zdftke:419` has no stratification
term on the diagonal, so an implicit-linearised selection would have had its
sink silently DELETED rather than moved — a computed `buoy_sink_rate` that
nothing reads.  Every shipped literal card already selects `nemo_explicit`
(`nemo_testcase_recipe.py:129`, `experiments/dino.py:1234`), so this changes no
trajectory; it turns a silent wrong answer into a loud refusal.

## MEASUREMENT — THE YEAR (P3(c)), and PHASE 0

**P3(c) HELD.**  All four from-rest members — the unperturbed control `seed 0`
and three `1e-10 K` IC perturbations — completed the full 360 days / 2160 steps
on the certified GYRE card, `1743`-`1872` s of wall each on CPU/fp64.  The
harness aborts on the first non-finite value at every 30-day snapshot, so
twelve snapshots per member reaching disk IS the finiteness statement.  The old
code could not reach step 48.

`--score-phase0` then wrote `year_fromrest/phase0_floor.json` at commit
`95cfcc38f18e` with a CLEAN worktree, and its own gates pass:

```
STATUS HELD
P1_floor_360_below_1e-3_K : 1.6588e-07 K against 1e-3     HELD
P2_floor_growth_below_1e3 : 332.17     against 1e3        HELD
vacuity_gate.phase1_may_run : true      (floor positive on every scored day)
floor_shape                 : FLOOR_GROWING
```

The ensemble spread, which IS the noise floor the later model-vs-model gap gets
judged against (max over the six within-ensemble pairwise distances, i.e. the
sample range at n=4):

| row | unit | day 30 | day 90 | day 180 | day 360 |
|---|---|---:|---:|---:|---:|
| T3D | K | 4.9938e-10 | 1.7339e-09 | 2.3139e-08 | 1.6588e-07 |
| S3D | g/kg | 2.1621e-10 | 2.9076e-10 | 3.0927e-09 | 1.3991e-08 |
| SST | K | 5.7524e-10 | 2.1935e-09 | 1.9190e-08 | 6.3876e-08 |
| SSH | m | 1.0332e-11 | 4.7240e-11 | 9.6197e-11 | 1.7854e-09 |
| T3D 0-100 m | K | 9.4695e-10 | 2.2096e-09 | 4.2724e-08 | 3.0912e-07 |
| T3D 100-1000 m | K | 1.8142e-10 | 2.2602e-09 | 1.3680e-08 | 8.6988e-08 |
| T3D >1000 m | K | 8.2206e-11 | 6.9514e-11 | 8.0291e-11 | 2.3389e-09 |
| PSI max | Sv | 2.6913e-10 | 9.9348e-10 | 4.1874e-10 | 1.1094e-07 |
| PSI min | Sv | 1.2792e-10 | 4.4844e-10 | 4.0310e-10 | 5.9132e-08 |
| QNET | W | 1.0294e+04 | 2.0874e+04 | 1.4077e+06 | 8.4027e+05 |

The floor GROWS by 332x over the year rather than collapsing, which is the case
the harness wants: a collapsing floor would make `gap/(2*floor)` a statement
about the perturbation decaying rather than about fidelity.  Its relative
standard error at n=4 is `0.41`, so these are order-of-magnitude figures, not
three-digit ones.  `repro_floor` (the same-binary irreproducibility floor,
which is a different quantity) stays UNMEASURED.

This is a legoESM-only number.  **NEMO's side of the year does not exist yet**,
so NOTHING here is a fidelity statement; Phase 0's only job is to say whether
the floor can see anything at all, and it says yes.

## RULE 12 — every card that executes the changed statement

| card | executes the literal TKE solve? | measurement | verdict |
|---|---|---|---|
| GYRE-zco (this card) | YES — `tke_matrix_evaluation`/`tke_solver_evaluation` both `nemo_literal`, `prognostic=True` (printed from the built card) |  corner series + 250-step census + the full 360-day year, all above | FIXED — the runaway is gone and the deepest row sits on `rn_emin` |
| LOCK_EXCHANGE-zco | NO | the built card's `model_config.physics` is `None` — there is no vertical-mixing block to select a scheme in. NEMO agrees: `tests/LOCK_EXCHANGE_OMIP_L1_P3_R33ZDF/EXP00/namelist_cfg:131` sets `ln_zdfcst = .true.` | UNAFFECTED BY CONSTRUCTION, both sides |
| OVERFLOW-zps | NO | same; NEMO `tests/OVERFLOW_OMIP_L1/EXP00/namelist_cfg:129` `ln_zdfcst = .true.` | UNAFFECTED BY CONSTRUCTION, both sides |
| DINO (`nemo_dino_kamm`) | YES — `experiments/dino.py:1248-1249` selects both literal evaluations | see below; the `zdf_chain_walk` logs are byte-identical before and after, and that fact is WORTH NOTHING on its own | UNMEASURED against NEMO; argued UNCHANGED from the code |
| ORCA2 | UNKNOWN-with-spec | no ORCA2 card exists in `nemo_testcase_recipe.py` (the builders are LOCK_EXCHANGE-zco, OVERFLOW-zps, GYRE-zco) | UNMEASURED. Spec: an ORCA2 card selecting `tke_matrix_evaluation="nemo_literal"` on real bathymetry would exercise BOTH the deepest-row row fixed here AND the `bottom_level == N` pin registered below |

### The DINO row, argued properly

A diff reviewer was right that quoting the byte-identical `zdf_chain_walk` logs
as evidence is a tautology dressed as a measurement: that probe feeds every
stage NEMO's OWN dumps, and its `EN (tridiagonal solve)` stage reports `n = 0,
UNMEASURED` because NEMO's SAVEd previous-step `dissl` is not dumped for that
run and the probe refuses to guess one.  No stage in it consumes the changed
solver, so it COULD NOT have moved.  The identity is recorded as a
not-broken-anything check, not as fidelity evidence.

The real argument for DINO is in the code, and it is about the bottom pin.
DINO sets `bottom_tke_bc = True` (`experiments/dino.py`), so
`tke.py`'s bottom-Dirichlet block runs AFTER the matrix assembly and forces
`a = 0`, `c = 0`, `diag = 1`, `rhs = e_bd` on the pinned row.  On a FULL-DEPTH
column the pin index clips to the deepest carried row, so that row is pinned
both before and after this change and its value is unchanged.  On a SHALLOWER
column the deepest carried row is below the seafloor, `w_active` is 0 there, so
`zcof`, `literal_up`, `literal_lw` and the `dissl` term are all zero and the
newly-built diagonal evaluates to exactly `1.0` — the constant it was hard-set
to before.  Either way DINO's trajectory is unchanged by construction.  That is
an ARGUMENT, not a measurement; DINO's own year was not re-run this round.

## REGISTER — rows this round MOVED or EXPOSED, none reverted

1. **`bottom_tke_bc` on a full-depth column pins a row NEMO solves.**  When
   `bottom_level == N` (a column wet to the last carried interface, i.e. NEMO
   `mbkt = jpkm1`), NEMO's held row is `mbkt+1 = jpk`, which legoESM does not
   carry; `tke.py:1517-1522` clips the pin index to `N-1` and so pins NEMO's
   `jpkm1` instead.  Behaviour is IDENTICAL before and after this change (the
   deepest row's diagonal was `1.0` either way and the pin overwrites the RHS),
   so nothing moved — but the row is now the only place the old convention
   survives.  GYRE has `bottom_tke_bc = False` so it is DEAD here; DINO sets it
   `True` (`experiments/dino.py`), so it is LIVE there.  NOT fixed this round:
   it is a second variable and GYRE cannot measure it.
2. **`rn_mxl0` is 4x too large on the GYRE identity card.**  Found by review B,
   verified in source: `zdftke:827-831`, `IF( ln_mxl0 ) rn_mxl0 = rmxl_min`,
   with `rmxl_min = 1.e-6/(rn_ediff*SQRT(rn_emin)) = 0.01` (`zdftke:815`).  The
   namelist's `rn_mxl0 = 0.04` is OVERWRITTEN by NEMO and never used.  The
   GYRE card sets `mxl0_min_m = 0.04` (`nemo_testcase_recipe.py:131`); the DINO
   card already has this right and says so in its own comment
   (`experiments/dino.py`, `"tke_mxl0_min_m": 0.01, # NEMO rmxl_min (ln_mxl0
   overwrites rn_mxl0=0.04)`).  So this is a known-and-fixed thing the GYRE card
   missed.  NOT landed here: it is a second variable that would confound this
   round's Rule-12 measurement, and it is the next round's first item.
3. **The `nn_mxl=3` `ldown` seed is off by one carried row** — the same family
   of defect as the one fixed here, in `compute_mixing_lengths`.  NEMO seeds the
   bottom-up sweep from `zmxlm(jpk) = rmxl_min` (`zdftke:593-594`, a row the raw
   fill `jk=2..jpkm1` never overwrites); legoESM seeds from the raw buoyancy
   length at its deepest carried row.  MEASURED, not inferred, by the campaign's
   own DINO chain walk: `zmxlm` err_norm `0.0885`, `max_diff` 617.5 m, on a
   near-neutral deep column.  Already escalated by that probe; NOT this round's
   change.  Row **N9** of the alignment table is therefore no longer UNMEASURED:
   it is DEBT.

## Retractions

1. **P1's headline is RETRACTED and re-headed.**  I named `tke.py:965` as the
   owning statement.  Review A proved that changing it alone moves nothing at
   the runaway cell — with `a[-1]=0` and `b[-1]=1` the solver's terminal row
   equals its raw right-hand side for either value of `jpkm1` — and would push
   the bad value up one level.  The proximate owner is the trailing identity
   diagonal, `tke.py:1457-1459`.  All three statements are one transcription.
2. **P4 is RETRACTED before measurement.**  See P4 above: its arithmetic was
   wrong (`e3t*e3w` is `9.0e4`, not `2.5e5`) and, more importantly, its
   mechanism was wrong — the post-solve `rn_emin` floor clamps both arms to the
   same value at `kt=1`.
3. Row **N9** was written as "cannot change the runaway's SIGN, so it cannot own
   the blow-up".  That remains true, and review B independently exonerated it
   for GYRE by showing NEMO's own ladder predicts the measured `L = 306.5 m`.
   But its "NOT RE-CERTIFIED" disposition understated what was already known:
   the campaign's DINO probe had already MEASURED it wrong.  Corrected above.

## ASKED / UNASKED

| choice | status |
|---|---|
| transcribe `zdftke:419` / `:412-418` / `:468` into the deepest carried TKE row | not a choice — NEMO's statements, cited |
| pad the upper viscosity neighbour at `jpkm1` with ZERO | not a choice — `zdfphy.f90:226-228` + `tke_avn`'s `jk=1,jpkm1` loop make `p_avm(jpk)` zero for the whole run |
| keep `c_diff[N-1] = 0` while keeping `-zzd_up(jpkm1)` in the diagonal | not a choice — `zdftke:419` vs `:468` |
| REFUSE `tke_buoyancy_sink != "nemo_explicit"` under the literal matrix | UNASKED, offered for revert. It changes no shipped trajectory (both literal cards already select `nemo_explicit`); it converts a silent deletion of the stratification sink into a loud error. Say the word and it comes out |
| add `--census-corner J,I,K` to the committed year harness | not a choice — a diagnostic print, reads carried state, computes nothing the model uses |
| do NOT land `rn_mxl0 = rmxl_min` this round | UNASKED, and the reason is the controlled-comparison rule: it is a second variable. It is a CONFIRMED, cited defect and the next round's first item |
| do NOT land the `bottom_level == N` pin correction this round | UNASKED, same reason; it is dead on GYRE and live on DINO, and DINO cannot measure it from here |

## Evidence

Under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tke_runaway/`
(`artifacts.sha256`):

```
2350835f997f2af9622fdc51bc463ab70d4578c5d454df7740bdc57c1ed0871d  corner_before.log
e8f73d18c403f908585f282263944a78e1391ba7312fa4802d8e695a186c2683  corner_after.log
f01b97591a1e463cd723dae387acb1938c0aa1c0ae351712d570d29c467cf939  dino_chain_before.log
f01b97591a1e463cd723dae387acb1938c0aa1c0ae351712d570d29c467cf939  dino_chain_after.log
7b9ce93e483f66dbad7487d6de7a2c423b710b96ca4c71dd547ab7f55acaaf80  operator_level.log
235bf8b21921b11d47eb03eb50aeca57f0ee2e465f76fa867933f387d5cb5020  ladder_compare.log
b61a8b9a7a0a9618a351a0439f36710c286e0783c1fa917b689730391a8389ab  ladder_before.npz
317ef3633ec3091310262d60298fd79246363ef5c0c633723534e68d2d5e8157  ladder_after.npz
ff2ca3eaff13516530d6536585bdcd7060db78af1798d883c70fc40bda8c83cd  phase0_score.log
3bd6712968cabf7c2aadc304b7c3fabda7548a37b75250d67af2e73f007427bd  tke_suite_final2.log
133de706738eb49770817df9ef1af7e46a7a19ee6185c96c5d1ec61d81801958  nemo_ladder_after.json
3508b5c452d09211eeb4d87bfaef5ec9610613ad9f9caba74a9659535294ebb2  ../year_fromrest/phase0_floor.json
6f8565649d7171bfd41020b2482ec9f010f9bd330a5c12a842e9f579f5d332e8  ../round48/round48_GYRE_kt1_2.json
```

The two DINO logs are BYTE-IDENTICAL, which is the Rule-12 row for that card
stated as a digest rather than as an opinion.

## Instrument obstacles met this round, recorded not worked around

* The GYRE `kt=1..10` gate refuses to run against its DEFAULT oracle root:
  `gyre_kt1_10_stage2_terms/oracle_stage_kt00000001_s1.bin` hashes
  `35e6892b799aeaf8d06d4affcd71b5ba0c71dc41bc0e8970c033459c46cd1402` against the
  pinned `ce25b004e7e8289b6e803263f895576981ce22516ccddfbd85d7be5ce5bcaedc`, and
  it stops at that identity check before the trajectory arm, `--trajectory-only`
  included.  **That was an obstacle, not a blocker, and an earlier draft of this
  receipt wrongly reported the ladder as unrunnable.**  The campaign's recent
  rounds point the gate at `round19_oracle_v2_external`, where every pin matches;
  the ladder is measured in its own section above.  The default root's mismatch
  remains an oracle-artifact registry problem that is not mine to fix (NEMO is
  read-only here).
* `tests/ocean/unit/test_tke_veros_dz_slots.py::test_solver_dz_cell_without_dz_surface_raises`
  is RED, and was red before this change: the raise it greps for is
  byte-identical at `HEAD` (`git show HEAD:...tke.py`, "dz_cell requires
  dz_surface too") and the test's regex says "dz_cell and dz_surface".  A
  message/regex mismatch in a guard this round did not touch.

## Review round — what the two diff reviews found, and what was done

Two fresh, independent Claude reviewers, one per angle.  **Codex was
unavailable** (another agent holds it, working the lateral-viscosity operators
on this same branch) and **GLM was unavailable**, so both reviews are Claude
agents with no shared context, each given a different attack surface.

| finding | disposition |
|---|---|
| **The new tests could not catch a wrong dissipation treatment.** Two mutations survived the WHOLE repo: dropping NEMO's trailing `* wmask` from either half of the split (every literal-matrix case passes `w_active = ones`), and ANY dissipation mutation confined to the `jpkm1` row (the one hand-computed matrix case pins that row with `bottom_dirichlet`, which overwrites its diagonal and RHS) | FIXED. `test_nemo_literal_matrix_pins_every_row_including_jpkm1` writes out all four rows of `a`/`b`/`c`/`rhs` by hand with NO bottom pin, `w_active = [1,0,1,1]` and the DEEPEST row WET, and carries four explicit non-vacuity assertions for the mutations that used to survive |
| `dz_half` and `w_active` lengths were unguarded, and the literal assembly now reads one more row of each; a short operand TRUNCATES silently in JAX | FIXED, two guards beside the existing `nemo_e3t` one |
| the `rhs[..., jpkm1+1:]` tail is now always empty and its comment is stale | FIXED — the tail is gone and an assertion says the array must end at `jpkm1` |
| the 1.5/0.5 weights, the carried-`dissl` operand and the add-back's sign are already pinned by `test_nemo_literal_matrix_matches_hand_computed_source_order` | confirmed, no action |
| `c_diff`'s trailing zero is inert (the solver reads `c[..., :jpkm1]`); the load-bearing half is `-literal_up[N-1]` in the diagonal | confirmed |
| `jpkm1 = L-1` verified against a dense solve at N = 1, 2, 3, 5, 29, max error 2.2e-16 | confirmed independently |
| **the commit says the model "died at step 48"; the log says 47** | corrected above — 47 is where the census catches it, 48 is the model's own abort |
| **"100.0000% from step 41" is off by one** — kt=41 is `1.000002` | corrected above |
| **"the predicted 16.4437"** understates the agreement, which is 1 part in 1e7 against the measured `L` | corrected above |
| **`1.979e-03` is the kt=240 value, not the run maximum** (`2.852e-03` at kt=10) | corrected above |
| **"stays in the ranges the healthy first 46 steps had" compares a 250-step window to a 46-step one** | WITHDRAWN, replaced by the matched-`kt=40` table above |
| **the byte-identical DINO logs are a tautology** — that probe's `EN` stage is UNMEASURED, so no stage consumes the changed solver | ACCEPTED. The Rule-12 row now says UNMEASURED and the real code argument is written out separately |
| **register item 3 (the `ldown` seed) feeds the very diagonal this change activates** | ACCEPTED, the coupling is now stated |
| **P3(c) had no result and the table still said FIXED** | the year is reported below, and "FIXED" is now defined as "the runaway is gone and the card completes", explicitly NOT "at the exact bar" |
| the guard added from the claim review turned 5 existing literal tests red | those tests set `K_H_old` or `N2` to ZERO, so `buoy_source = -K_H*N2 = 0` under either branch and `buoy_sink_rate` is unread by the literal diagonal. They now select `nemo_explicit`; a provable numerical no-op, and the assertions are untouched |

Pre-existing reds, each shown pre-existing before being dismissed:
`test_tke_veros_dz_slots.py::test_solver_dz_cell_without_dz_surface_raises`
(the raise it greps for is byte-identical at `HEAD`), and three in
`test_implicit_vmix_face_control_volume.py`, which fail inside a test-side
monkeypatch calling `np.asarray` on a traced array under
`ocean_model_latlon_cgrid._step_jitted` — a file this round does not touch, in
a test file with zero references to the TKE closure.

## HANDOFF — the NEMO acquisition, for the operator to run

Phase 0 is complete and `year_fromrest/phase0_floor.json` exists, so the
acquisition script's own gate is satisfied.  The agent does not run NEMO,
`makenemo` or `mpirun`; these are the operator's commands.

Phase 0, already run (reproduces the table above; no NEMO time):

```
cd /tmp/codex-gyre
export PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
H=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_fromrest.py
for s in 0 1 2 3; do /home/dbalwada/legoESM/.venv/bin/python $H --member $s; done
/home/dbalwada/legoESM/.venv/bin/python $H --score-phase0
```

Phase 1, NEMO — ONE command, which builds one patched config and runs five year
members (`nn_pert_seed = 0..3` plus a pristine control on the UNPATCHED
certified binary):

```
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_fromrest_members/run.sh
```

It copies `GYRE_OMIP_L2_P3_SM_R41ADVSP` to `GYRE_OMIP_L2_P3_SM_YRPERT`, applies
the two additive `MY_SRC` patches that add `nn_pert_seed`, checks the patches
removed only the one `NAMELIST` line, verifies `nn_pert_seed` reached the
COMPILED `ppsrc`, refuses if any namelist row outside
`nn_itend`/`nn_stock`/`nn_write`/`nn_pert_seed` moved, runs
`mpirun -np 1 ./nemo` per member at `nn_itend = 2160`, `nn_stock = 180`,
requires every seed-0 restart to be BYTE-IDENTICAL to the pristine run's,
refuses if any two members come out bit-identical, and writes
`nemo_year_fromrest_restarts.sha256`.  It prints
`GYRE_YEAR_FROMREST_NEMO_READY <dir>` on success.

Then, back on the agent side:

```
/home/dbalwada/legoESM/.venv/bin/python $H --score
/home/dbalwada/legoESM/.venv/bin/python $H --figures
```

STOPPING HERE for the acquisition, as the round's rules require.

## THE AGAINST-NEMO LADDER — measured, and UNRECONCILED (Rule 1e)

The gate that compares `kt=1..10` against NEMO's record DOES run, once pointed
at the oracle root the campaign actually uses.  My earlier "blocked" note was
wrong in its conclusion though right in its facts: the DEFAULT root's stage-2
artifact hash does not match the pin, but `round19_oracle_v2_external` does, and
that is the root every recent round has used.  Corrected command:

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_phase3_gate.py \
  --oracle-root       /data/abyssal/.../phase3/round19_oracle_v2_external \
  --stage2-oracle-root ... --stage3-oracle-root ... \
  --max-step 10 --trajectory-only --without-oracle-ene-coefficients
```

AFTER, at commit `20974116a92b`, against the round-48 record
`round48/round48_GYRE_kt1_2.json` (BEFORE, commit `fcacdf16d189`, SAME gate,
SAME oracle root, same flags):

| field, max abs residual vs NEMO | kt=1 before | kt=1 after | kt=2 before | kt=2 after |
|---|---:|---:|---:|---:|
| u | 0.0 | 0.0 | 2.7478404751243857e-12 | 2.7478404751243857e-12 |
| v | 0.0 | 0.0 | 3.305560306813421e-12 | 3.305560306813421e-12 |
| T | 0.0 | 0.0 | 1.4210854715202004e-14 | 1.4210854715202004e-14 |
| S | 0.0 | 0.0 | 2.1316282072803006e-14 | 2.1316282072803006e-14 |
| ssh | 0.0 | 0.0 | 4.336808689942018e-19 | 4.336808689942018e-19 |

Every field, both steps, BIT-IDENTICAL before and after; `first_over_bar` stays
`kt=2` on `u`/`v`, and `kt=1` is exact.  `kt=3..10` are `7e-4` to `5e-2 m/s`,
which is the campaign's known downstream divergence and is not compared here
because no matched BEFORE run past `kt=2` exists.

**AND THAT DISAGREES WITH MY OWN SELF-A/B**, which found `kt=2` moving (`u` by
`1.09e-10`, `T` by `5.5e-09`, TKE by `2.3e-07`).  Both cannot be describing the
same trajectory, so NEITHER is recorded as "the" ladder answer.

The reconciliation is NOT done, and the cause is already localised to a
CONFIGURATION difference between the two instruments, which is the trap this
campaign has hit before: the phase-3 gate does not run the card as built, it
runs

```
cfg = card.recipe.model_config._replace(
    freshwater_closure="real_freshwater", fix_eta_drift=True)
```

while the year harness steps `card.recipe.model_config` unmodified.  Two
different resolved configurations, so the two trajectories are not the same
run and their `kt=2` answers are not comparable.

**The discriminating measurement, named and not yet run:** step the year
harness's card with those two fields set the gate's way and re-take the
self-A/B; if `kt=2` then comes out bit-identical, the gate's row stands and the
self-A/B was measuring the other configuration.  Until that is run, what is
CONFIRMED is: `kt=1` is bit-identical under BOTH instruments and exact against
NEMO, and the gate's card is unchanged at `kt=1` and `kt=2` by this fix.
