# VORTEX round 18 (lane round 204) — the flux card's stage-1 producer, split and named

**ROUND_STATUS: HELD** — the instrument lands, no physics and no
configuration changes, and no NEMO run was needed.  Round 201's OPEN item 1
is CLOSED: the stage-1 right-hand side is now split into the two statements
that build it, and the first non-bit producer is named.

## The one-sentence result

**The flux card's stage-1 error is made inside the flux-form momentum
advection, `dyn_adv_up3` as called from `stprk3_stg.f90:316` — not by the
hydrostatic pressure gradient (`stp2d.f90:138`) and not by `dyn_vor`
(`:144`).**  legoESM's stage-1 right-hand side WITHOUT its advection content
agrees with NEMO's recorded pre-`dyn_adv` array to `2.710505e-20` **in the
depth-varying channel**, which is the arithmetic floor of the diagnostic; the
advection trend differs by `4.540459e-11` in that same channel, which is the
WHOLE of the completed right-hand side's error there.

**Said precisely, because the first draft of this sentence was refuted in
review.**  `dyn_adv`'s OWN OPERANDS are already non-bit when it runs — the
walk's `first_non_bit` field says `zfu`, not `adv` — so "the first non-bit
thing in stage 1" and "the owner" are not the same statement.  What makes the
operands innocent is a bound, now stated instead of assumed: on this mesh
`e1e2u = 9.0e8 m^2` and `e3u = 500 m`, both uniform, and `|u|` peaks at
`0.8723 m/s`, so a transport error of `1.862645e-09` can move the
flux-divergence trend by at most `3.6e-21` per face and `zFw`'s
`2.447726e-08` by at most `4.7e-20` — **ten and nine orders below the
measured `6.285649e-11`**, with only `O(4)` faces per cell to multiply it.
The owner is `dyn_adv_up3`'s own arithmetic.

Only the depth-varying channel can reach the stage output: the barotropic
replacement at `stprk3_stg.f90:409-421` sets each column's mean from the
external solve.  Every attribution below is made on that channel and says so.

## Why round 201 could not do this, read from the compiled source

`/data/abyssal/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/VORTEX_OMIP_L1_P3/BLD/ppsrc/nemo/`:

* `stp2d.f90:137-144` — `eos`, `dyn_hpg`, `dyn_ldf`, `dyn_vor` write the
  three-dimensional `Krhs` at `Kbb`.  This deck pins `ln_dynldf_OFF=.true.`
  (`EXP00/namelist_cfg:222`) so `dyn_ldf` contributes nothing, and
  `ln_dynvor_een=.true.` (`:193`): the pre-advection array is HPG + EEN
  COR/MET.
* `stp2d.f90:169-170` — `CASE( np_FLX_up3 )` calls
  `dyn_adv_up3( kt, Kbb, Kbb, uu, vv, Krhs, pUe=Ue_rhs, pVe=Ve_rhs )`,
  labelled "2D RHS only"; with `pUe` present `dynadv_up3.f90:201,288,349`
  and `:362-364` write `pUe`/`pVe` alone.  So NEMO's stage-1 `Krhs` carries
  NO advection at all.
* legoESM's own pre-stage array is its completed right-hand side, advection
  included.  The record's `base` row was therefore not like-for-like, and
  round 201 reported it without attribution.

## What this round built

One private WRITE-only hook, the companion of round 201's three:

| arm | publishes | NEMO boundary |
|---|---|---|
| `expose_stage1_momentum_rhs_split="pre_advection"` | legoESM's stage-1 right-hand side with its advection content removed | `Krhs` as `stp_2D` leaves it (`stp2d.f90:137-144`) |
| `expose_stage1_momentum_rhs_split="completed"` | the completed right-hand side from the SAME evaluation | after `stprk3_stg.f90:316` |

Default `""`; no card constructs it; the construction guard refuses it
alongside any other momentum exposure or the stage-1 override, because they
share the returned u/v slots; and it fails closed under
`adaptive_implicit_vertadv`, where the step-level tendency does not carry the
vertical advection the decomposition reports.  Both frames are substituted
into the returned diagnostic state only after the ordinary step completes.

**The arithmetic, stated so it can be checked.**  legoESM's completed stage-1
right-hand side is `du_dt_pert` plus two increments — the `zub` transport
operand and the stage ZAD operand — and BOTH increments are pure advection
(each is the difference of two otherwise identical tendency evaluations).
The vertical UP3 addend is `None` on this card because the deck leaves
`ln_zad_Aimp=.false.` (`cfgs/SHARED/namelist_ref:1177`, not overridden), so
the vertical advection lives inside the step-level tendency.  The published
pre-advection frame is therefore `du_dt_pert` minus the step-level advection
component of the SAME evaluation.

**A defect found and fixed inside the instrument, because it changed the
answer by an order of magnitude.**  In flux form legoESM writes the
horizontal `-div(transport (x) velocity)` trend into the ROTATION diagnostic
slot and zeroes the kinetic-energy gradient that would otherwise hold it, so
the published `advection_u` component (`-dKE_dx + Dterm + vertadv`) is NOT
the advective half on a flux-form card.  The first draft of this seam
subtracted only that, and the scored row came out at `9.195320e-05` — WORSE
than the un-split row it was meant to replace.  The operator-component bundle
now publishes `flux_form_hadv_u/v` apart, captured before the rotation
branches add Coriolis into the same slot, and the committed test pins the
property that caught it.

## The walk, with the split scored

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round200_flux_stage1.py`,
production-jitted step, NEMO's recorded kt=1 stage entry and recorded
barotropic quintuple (round 200's design: the barotropic solve is a known
separate owner and is held fixed on EVERY arm).  Evidence
`phase3/round204/flux_stage1_split.{json,log}`.  "depth-uniform" is the
largest column mean of the difference, "depth-varying" the largest departure
from it.

| boundary | citation | cells | max abs | depth-uniform | depth-varying |
|---|---|---:|---:|---:|---:|
| pre-stage RHS `u`, UNSPLIT (round 201) | `stp2d.f90:126-171` | 18997 | `1.370473e-05` | `2.790517e-06` | `1.091421e-05` |
| **pre-advection RHS `u`** | **`stp2d.f90:137-144,169-170`** | 36600 | `8.142169e-05` | `8.142169373e-05` | **`2.710505e-20`** |
| **pre-advection RHS `v`** | same | 36600 | `8.171075e-05` | `8.171075457e-05` | **`2.710505e-20`** |
| **advection trend `u`** | **`stprk3_stg.f90:316`** | 10614 | `6.285649e-11` | `1.745190e-11` | **`4.540459e-11`** |
| **advection trend `v`** | same | 12206 | `6.279942e-11` | `1.743606e-11` | **`4.536336e-11`** |
| completed RHS `u` (round 201) | `stprk3_stg.f90:316` | 36600 | `8.142171e-05` | `8.142170e-05` | `4.540459e-11` |
| completed RHS `v` (round 201) | same | 36600 | `8.171077e-05` | `8.171076e-05` | `4.536336e-11` |

**The split closes.**  The advection trend's depth-varying difference
`4.540458839058695e-11` reproduces the completed row's
`4.540458838715646e-11` to `7.6e-11` relative (`v`:
`4.536336143204694e-11` against `4.536336142336485e-11`, `1.9e-10`).  The
pre-advection row contributes `2.710505431213761e-20`, i.e. `6.0e-10` of the
trend's share.

**Why `2.71e-20` is a floor and not an agreement being overstated.**  The
statistic is the largest departure from a column mean of numbers whose size
is `8.14e-05`; one unit in the last place there is `1.4e-20`, so `2.7e-20` is
two of them.  The honest statement is that the pre-advection right-hand sides
agree **to the floor of the diagnostic's own arithmetic**, not that they are
bit-identical — they are not, by the `8.14e-05` depth-uniform offset, which
is the known `du_dt_pert = du_dt - F_slow` convention difference round 201
measured and proved immaterial (`stp2d.f90:176-186` cumulates NEMO's mean
into `Ue_rhs`; it does not subtract it from the three-dimensional array).

**The operands, and why they are not the owner.**  Round 200 measured the
stage advective transports that `dyn_adv` consumes: `zFu`/`zFv` differ by
`1.862645e-09` (one unit in the last place, `1.4e-16` relative, 118 and 128
of 36600 faces), `zFw` by `2.447726e-08` (`2.8e-13` relative) and `ww` by
`2.719696e-17` (the same `2.8e-13`).  They are non-bit, and they run before
`:316`, so the walk reports `zfu` as the first non-bit row of the whole
stage.  The bound in the headline is what separates them from the owner:
`|δF| · |u| / (e1e2u · e3u)` with the mesh's own numbers is `3.6e-21` for
`zFu` and `4.7e-20` for `zFw`, against a measured trend error of
`6.285649e-11`.  CONFIRMED by arithmetic on measured rows and on the card's
own metrics (`e1e2u` and `dz_ref` are uniform on this mesh, so there is no
worst-cell ambiguity); the one thing it assumes is that the trend is a flux
divergence over `O(4)` faces, which `dynadv_up3.f90:201-215` states.

**What the producer's size is.**  The advection trend differs by
`4.585e-06` of its own peak (`u`; `4.582e-06` for `v`).  That is far above
any rounding floor and far above the one-unit-in-the-last-place difference
round 200 measured in the transports `zFu`/`zFv` that feed it
(`1.862645e-09`, `1.4e-16` relative, 118/128 of 36600 faces).  So the trend's
disagreement is NOT inherited from its transport operands: it is made inside
`dyn_adv_up3`.  That sentence is CONFIRMED for the transports named and
PLAUSIBLE as an exclusion of every other operand, because `zFw`
(`2.800e-13` relative) was not separately propagated.

## Pre-registration: what held and what did not

The frozen file is `phase3/round204/prereg.md`
(`sha256 ef7a6be11eb0e81f3d7a61a8bee10be1a919214b1fc820109ae62ebfe001b139`),
written before any measurement.

* **P1 zero behaviour change — CONFIRMED.**  Gate table below.
* **P2 the seam is live — CONFIRMED.**  All four new rows VISIBLE as a
  DIFFERENCE against the unplanted run (`phase3/round204/split_plants.log`).
* **P3 the decomposition does not perturb the step — CONFIRMED.**  The
  completed right-hand side read back with `diagnose_momentum` on differs
  from the production-configured one in **0 of 36600 `u` cells and 0 of
  36600 `v` cells**; the walk REFUSES if it does not.
* **P4 the producer is the advection trend, not the pre-advection RHS —
  CONFIRMED**, and by a wider margin than predicted: P4 allowed the
  pre-advection row up to `1e-13` of the field's peak and it came in at the
  `2.7e-20` arithmetic floor.
* **P5 the split is a partition — REFUTED AS WRITTEN.**  P5 said the two
  halves put the completed right-hand side back "bit for bit"; adding and
  subtracting the same array is not an identity in floating point and it does
  not.  Measured: `6.617445e-24` (`u`), `0.0` (`v`) — the floor, not zero.
  P5's intent holds; its written falsifier fires, and it is recorded REFUTED
  rather than quietly rescored.

## Gates

| gate | result |
|---|---|
| `VORTEX-zco` certified registry vs round 203 | **PASS**, 50 rows, `max_worsening_ulps 0`, largest field move `0.0` row-scale ULPs — **0 of 50 rows move**, 0 violations, no row-status change, `first_over_bar` unchanged `{T,u,v,ssh} kt=2` |
| `VORTEX_VEC-zco` certified registry vs round 203 | **PASS**, 50 rows, `max_worsening_ulps 0`, largest field move `0.0` — **inert, 0 of 50**, `first_over_bar` unchanged `{u,v,ssh} kt=2` |
| cellwise two-ULP ratchet plants (flux card) | `worsen-3ulp` exit 1 (`max_worsening_ulps=3`), `at-bar-to-debt` exit 1 (`VORTEX-zco.kt1.before.S` AT-BAR→DEBT) — both red on the same 50-row pair the unplanted run passes at 0 ULP |
| GYRE certified kt=1..10 ladder vs round 203 | **PASS**, **954 rows, 0 move, 0 ULP**, `first_over_bar` unchanged `{T,S,u,v,ssh} kt=3` |
| GYRE 360-day from-rest year (note BW) | **byte-identical**, 360 of 360 snapshots equal to the round-203 certified arm, 0 differing, 0 missing |
| citation gate, `DEFAULT_RECEIPT` | **PASS** exit 0, 274 citations, `unmapped_citations []`, `map_entries_failing_audit []`, all nine self-test plants fired |
| focused battery | **44 passed in 229.32s** (`..._round204_stage1_split.py`, `..._round201_stage1_seam.py`, `test_nemo_testcase_receipt_citation_gate.py`) |
| generic NEMO-GYRE recipe gate | run by `land.sh` with the push gate |
| DINO month gate (`land.sh`, reference `2.053801168e-03` K, bar `2.244317642e-03`) | run by `land.sh`; line quoted in the ledger |

```
certified arm files: 360 | compared: 360 | differing: 0 | missing: 0
day030.npz certified 4e36c106403b495e round204 4e36c106403b495e
day240.npz certified a63befc30bf03b44 round204 a63befc30bf03b44
day360.npz certified dcb7bc46c8bc75bd round204 dcb7bc46c8bc75bd
```

The three certified day numbers therefore stand to every printed digit
(note BW): **day 30 `2.3432465132112266e-06`**, **day 240
`6.58170624837412e-05`**, **day 360 `5.407736527246344e-05` K**.  They are
carried by byte-identity of the snapshots they are computed from; the day-gap
scorer was not re-run, and that is said here rather than implied.

**Provenance of the two long gates, said rather than implied.**  The ladder
and the year were measured at `31481fae21c2`, before the review-fix commit.
The only later change to `packages/`+`src/` is COMMENT lines:
`git diff 31481fae21c2..HEAD -- packages/ src/` filtered to non-comment
changed lines is EMPTY.  Everything else in those commits is the walk script,
test docstrings, the citation map and this receipt.

Evidence: `phase3/round204/prereg.{md,sha256}`,
`flux_stage1_split.{json,log}`, `split_plants.log`,
`traj_VORTEX{,_VEC}-zco_after.{json,log}`, `ulp_VORTEX{,_VEC}-zco.json`,
`ulp_plant_{worsen-3ulp,at-bar-to-debt}.{json,log}`,
`gyre_ladder_after.{json,log}`, `gyre_ladder_compare.json`,
`gyre_year_r204.log`, `gyre_year_byte_identity.txt`,
`citations_default_receipt.json`, `focused_pytest.log`.

## Independent review

One fresh `code-reviewer` subagent on the diff, told to refute the claim.
**First verdict: DO NOT SHIP**, two blockers and three minors.  All accepted:

1. **The headline attribution skipped a step.**  `dyn_adv`'s operands
   `zFu`/`zFv`/`zFw`/`ww` are themselves non-bit and all execute before
   `:316`, so the walk's `first_non_bit` field says `zfu`.  The sentence in
   the first commit message is RETRACTED as written; the conclusion is
   re-stated above with the operand-to-trend bound that makes it true, and
   the retraction is recorded in the commit that fixes it.
2. **`base_noadv` was not flagged convention-sensitive** although it carries
   a LARGER depth-uniform offset (`8.1e-05`) than the `base` row that is
   flagged (`2.79e-06`).  That flag is the only thing that keeps a row from
   being eligible to be named owner, so a non-bit `8.1e-05` row was eligible
   and only the accident of ordering hid it.  Flagged; the walk re-run.
3. The `2.7e-20` number must carry "in the depth-varying channel" every time
   it appears, because the row's own `max_abs` is `8.14e-05`.  Done.
4. The default-changes-nothing test covers the field's presence, not the new
   code behind it.  Its docstring now says so and points at the external
   registries, which are what actually carry P1.
5. The advection-only perturbation control exercises the flux-form
   HORIZONTAL term only.  Its docstring now says so.

The reviewer independently confirmed what the round rests on: the two
post-base stage addends reach only the flux-form horizontal advection and
only `dyn_zad`'s operands respectively, so both are purely advective; the
third addend is `None` here and the new guard covers the case where it would
not be; `flux_form_hadv` is captured before any rotation branch, stays zero
on vector-invariant and WENO cards, and cannot double-count `advection_u`
because the kinetic-energy gradient is explicitly zeroed in flux form; no
existing consumer reads the two new keys; the P3 control is real and passes;
and there is no hidden configuration choice or changed default.

## Non-vacuity

* **Seam control, inside the walk.**  Every row is read out of a state slot
  the ordinary step also fills, so an inert hook would hand the walk the
  plain output and the row would still score.  The walk runs the plain step
  once and refuses unless every exposed slot differs from it; both new
  exposures pass it.
* **Plants as a DIFFERENCE against the unplanted run** (round 200's lesson):
  `base_noadv.u`, `base_noadv.v`, `advtrend.u`, `advtrend.v` — all four
  VISIBLE (`phase3/round204/split_plants.log`).
* **The P3 control is a refusal, not a report**: the walk raises if the
  decomposition moves the completed right-hand side in a single cell, and the
  same property is pinned by a committed test.
* **The order-of-magnitude control that actually caught a defect**: the
  committed test requires that what the split removes have DEPTH STRUCTURE
  comparable to its own size.  The first draft of the seam — which removed
  only the vertical and D-term halves — fails it, and produced a scored row
  of `9.195320e-05` instead of `8.142169e-05`.
* **An advection-only perturbation must miss the published frame**: bending
  the UP3 upwind-curvature transport-sign selector moves the completed
  right-hand side with depth structure and moves the pre-advection frame by
  at most one number per column (the depth mean the stage array has already
  subtracted).  A frame that still carried advection fails this.
* **Construction-guard refusals** are parametrised over every ambiguous
  combination, including the two illegal strings.
* **Citation re-anchor was mechanical**: 76 spans re-anchored from a difflib
  old→new line map over the two edited model files
  (`ocean_model_latlon_cgrid.py` 15073→15144 lines,
  `ocean_pe_latlon_cgrid.py` 6072→6091), never an eyeballed delta; the gate's
  own extent audit then caught one span whose pinned LINE COUNT the re-anchor
  does not move, and that was fixed in its own commit.

## Choices made this round

| choice | ASKED / UNASKED |
|---|---|
| Build the split as a third stage-1 hook rather than relaxing the stage-1 refusal in `expose_momentum_operator` | mechanical; that refusal exists so a stage cannot be scored under another's name, and the stage-1 advection is assembled from three places, not one |
| Define the removed half as the step-level advection component plus the two stage increments | forced: those increments ARE the advection the stage adds, and the deck's `ln_zad_Aimp=.false.` makes the vertical addend `None` |
| Publish `flux_form_hadv_u/v` as a new operator component | required: without it the removed half is wrong on every flux-form card, measured |
| Fail closed under `adaptive_implicit_vertadv` instead of handling it | the decomposition reports a vertical term the tendency does not carry there; no card on this lane needs it |
| Score the split on the depth-varying channel only | forced by `stprk3_stg.f90:409-421`; stated in the pre-registration before measuring |

**UNASKED list: empty.**  No default, scheme, bound, threshold, deck value or
carried state was changed; nothing the NEMO deck does not pin was selected.

## OPEN, in order

1. **Inside `dyn_adv_up3`: which of its three parts owns the `4.585e-06`
   relative disagreement.**  The compiled routine writes the horizontal
   trend from the T-point UP3 fluxes (`dynadv_up3.f90:166-215`), a vertical
   flux block (`:239-358`) and a final metric division; legoESM assembles the
   horizontal half in its flux-form branch and the vertical half separately.
   The record carries only the TOTAL (`adv - base`), so separating them needs
   either a per-part `dyn_adv` record on the flux build (a NEMO run) or a
   one-variable substitution of legoESM's own halves against the total.  The
   substitution is the cheaper first move and needs no NEMO run.
2. The transports' own one-unit-in-the-last-place difference
   (`stprk3_stg.f90:276-277`): split `zub` (`:270`) from the
   `e3t_1d*(1+r3u(Kmm)*umask)` face thickness; neither is recorded.
3. The depth-uniform `8.14e-05` difference between the two codes' stage-1
   right-hand sides — immaterial to the trajectory (measured, round 201), but
   a real difference in what the two hold under one name.
4. The `fmask`-versus-`fe3mask` `r3f` factor, inert only at `rn_shlat=0`.
5. Decision 74's 30/15/10-km ladder, once the flux card's kt=2 rows are at
   the bar or proven harness floor.
