# NEMO testcase Lane 4 — ORCA2 card round 11 preregistration

Date: 2026-09-23

Parent: `c28ac176c51ea090913bcabac7517fd8dad1bda8`

Status: **PREREGISTERED BEFORE ANY ROUND-11 MEASUREMENT.**

Two statements, in this order, on the ocean-only `orca2_vector_een_c2` card.

**A.** The WHOLE lateral momentum viscosity operator — not only the extra
vertex mask round 9 counted — gated against NEMO's compiled `zwf`/`zwt` loops
on the record's own coefficients, metrics, thicknesses and kt=1 velocities.
No new NEMO run: every oracle number is recomputed from recorded inputs with
round 8's technique.

**B.** Round 10's OPEN item 1: the non-finite buoyancy-frequency divisor that
refuses the production step at kt=1.  The discriminator is one line and was
never run.

The six-entry sea-ice registry is frozen and out of scope.  Decision 52's
labels are binding: every number is either **given NEMO's entry** or
**independent**, never mixed in one table without its label.

## What the record fixes, and what nothing in this round may choose

Read from the record's own resolved configuration, not from a deck comment.
The record is the pinned ORCA1-ice reference run
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.

| resolved setting | value | where it is printed |
|---|---|---|
| lateral viscosity operator family | div-rot, `nn_dynldf_typ = 0`, laplacian, iso-level | run `ocean.output:1176-1180,1193,1195` |
| viscosity coefficient source | `ahmt_3d`/`ahmf_3d` read whole from `eddy_viscosity_3D.nc` (`nn_ahm_ijk_t = -30`) | run `ocean.output:1184,1197,1200-1201` |
| lateral momentum boundary condition | no-slip, `rn_shlat = 2.0` | run `ocean.output:339` |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output:217` |

No selector default, tunable, threshold, resolution, timestep, carried state
or data source moves in this round.  The GYRE card resolves `rn_shlat = 0` and
builds its coefficient from the metric formula; it must stay bit-identical.

## Statement A — the whole `dyn_ldf` operator

NEMO's resolved arm is `dynldf_lev_lap`'s `np_typ_rot` branch, called from
Runge-Kutta **stage three** with `Kbb = Nbb` and `Kmm = Nnn`
(`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:493`;
the dispatch is `dynldf.f90:85`).  The compiled loops are
`dynldf_lev.f90:121-141` and read, statement by statement:

```
zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk) * ( e3f_3d(ji-1,jj-1,jk)*(1+r3f(ji-1,jj-1)*fe3mask(ji-1,jj-1,jk)) )
   * r1_e1e2f(ji-1,jj-1)
   * ( ( e2v(ji,jj-1)*pv(ji,jj-1,jk,Kbb) - e2v(ji-1,jj-1)*pv(ji-1,jj-1,jk,Kbb) )
     - ( e1u(ji-1,jj)*pu(ji-1,jj,jk,Kbb) - e1u(ji-1,jj-1)*pu(ji-1,jj-1,jk,Kbb) ) )   ! ahmf already * by fmask
zwt(ji,jj)     = ahmt(ji,jj,jk) * r1_e1e2t(ji,jj) / ( e3t_3d(ji,jj,jk)*(1+r3t(ji,jj,Kbb)*tmask(ji,jj,jk)) )
   * ( ( e2u(ji,jj)*e3u_live(ji,jj,jk,Kbb)*pu(ji,jj,jk,Kbb) - e2u(ji-1,jj)*e3u_live(ji-1,jj,jk,Kbb)*pu(ji-1,jj,jk,Kbb) )
     + ( e1v(ji,jj)*e3v_live(ji,jj,jk,Kbb)*pv(ji,jj,jk,Kbb) - e1v(ji,jj-1)*e3v_live(ji,jj-1,jk,Kbb)*pv(ji,jj-1,jk,Kbb) ) )  ! ahmt already * by tmask
pu(Krhs) += umask * ( - ( zwf(ji,jj) - zwf(ji,jj-1) ) * r1_e2u / e3u_live(ji,jj,jk,Kmm)
                      + ( zwt(ji+1,jj) - zwt(ji,jj) ) * r1_e1u )
pv(Krhs) += vmask * ( + ( zwf(ji,jj) - zwf(ji-1,jj) ) * r1_e1v / e3v_live(ji,jj,jk,Kmm)
                      + ( zwt(ji,jj+1) - zwt(ji,jj) ) * r1_e2v )
```

(`dynldf_lev.f90:123`, `:127-129`, `:133-136`, `:137-140`).  `r3f` is the
area-weighted four-cell average of `ssh(Kbb)` (`domqco.f90:225`, and its
Runge-Kutta twin `:281`).

**NOTE, and a correction the round must settle.**  Round 10's section 6
described legoESM's operator as applying NEITHER the thickness weighting nor
the outer face-thickness division, citing the docstring of
`nemo_ldf_lap_viscosity_cgrid`.  The ORCA2 card does not call that function.
The shared NEMO-identity base sets `lateral_viscosity_operator="nemo_div_curl"`
AND `lateral_viscosity_e3_weighting="nemo_e3"`
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:308-309`), so
both ORCA2 and GYRE run `nemo_ldf_lap_viscosity_e3_cgrid`
(`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3440`), which
DOES carry a thickness weighting.  Whether it carries NEMO's thicknesses is a
MEASUREMENT this round makes, not a claim it inherits.

## Statement B — the non-finite buoyancy divisor at kt=1

The production step refuses inside the implicit vertical mixing: the raw-mesh
guard at `packages/ocean/legoesm/ocean/eos.py:738-742` requires a finite,
positive `e3w`.  That `e3w` is built by
`nemo_bn2_live_geometry` (`packages/ocean/legoesm/ocean/eos.py:831-849`) as the
raw `e3w_0` times the stretch `max(1 + ssh/ht_0, 1e-6)`
(`packages/ocean/legoesm/ocean/eos.py:906-967`), threaded from
`packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py:1253`.
The dry-column divide is already guarded there, so a non-finite divisor needs a
non-finite (or astronomically large) `ssh` or `ht_0`.  The discriminator prints
the sea surface and the column depth that the implicit vertical mixing actually
receives, and splits wet from dry.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R11-P1 | The ORCA2 card executes `nemo_ldf_lap_viscosity_e3_cgrid` (not its e3-free sibling), through `_bc_horizontal_viscosity`, with `thickness_operands=None` — so production takes the ALGEBRAIC branch and builds every thickness by the min rule from one time level. | The dispatch reads as stated and no production call site supplies `ldf_thickness_operands`. | Either the e3-free sibling runs, or production supplies the operands. |
| R11-P2 | Run on the record's own coefficients, metrics, masks, thicknesses and kt=1 stage-three velocities, legoESM's operator DIFFERS from the compiled `zwf`/`zwt` loops on a large fraction of owned wet cells. | A non-zero unequal count on both `u` and `v`. | Zero unequal, which would refute round 9's and round 10's entire premise. |
| R11-P3 | The DOUBLE MASK is a real, separable term: rebuilding the oracle with legoESM's extra zero/one vertex mask applied to `ahmf` changes the F-point contribution on the 48,287 cells round 9 counted, and on no others. | The mask-only ablation moves exactly the cells where NEMO's `ahmf` is non-zero and the vertex mask is zero (and their two RHS neighbours). | A different cell set, or a count that does not reproduce. |
| R11-P4 | The THICKNESS WEIGHTING is a second, separable and LARGER term: legoESM's min-rule `e3f`/`e3u`/`e3v` differ from NEMO's stored `e3f_0`/`e3u_0`/`e3v_0` on the partial-cell / topographic-step columns, and the resulting tendency difference exceeds the mask-only one. | Both ablations measured separately; the thickness one is larger. | The thickness fields agree bitwise, or its effect is the smaller of the two. |
| R11-P5 | The mask fix is scoped to ORCA2 alone: dropping the second masking ONLY where NEMO's mask is already inside the coefficient (`lateral_viscosity_coefficient_source="nemo_ahm_3d_file"`, which only the ORCA2 card selects) leaves GYRE, DINO, the generic cards and the tank cases bit-identical. | GYRE base-vs-tip: 0 differing rows, array-equal residuals, byte-identical 30-day snapshots, day-30 digest `14a7e64b4512860e`; DINO/generic/tank suites unchanged. | Any movement on any of them. |
| R11-P6 | The thickness fix CANNOT be scoped that way — GYRE runs the same `nemo_e3` weighting through the same operator — so it is a GYRE landing and becomes DECISION_NEEDED rather than a round-11 landing. | GYRE selects `nemo_e3` and its thicknesses would move. | GYRE is provably unaffected, in which case it may land here. |
| R11-P7 | The quantity that is non-finite inside the implicit vertical mixing at kt=1 is the SEA-SURFACE HEIGHT (round 10 left this PLAUSIBLE). | The discriminator reports a non-zero non-finite count in the sea surface the mixing receives, and zero in the column depth. | The sea surface is finite, in which case the column depth or the raw mesh field owns it and round 10's PLAUSIBLE label is REFUTED. |
| R11-P8 | That non-finite sea surface is produced INSIDE the step (the step-entry sea surface is the record's own and is finite), so its first producer is a Runge-Kutta stage statement, not the entry bridge. | The entry sea surface is finite and a stage output is not. | The entry sea surface already carries it. |
| R11-P9 | With the round's landing in place the ORCA2 ladder does NOT reach kt=10, because statement B is a separate defect from statement A. | The ladder still stops before kt=10. | It reaches kt=10, in which case the magnitude is registered. |

Failed predictions stay in the receipt as **REFUTED** and are never quietly
dropped.

## Controls and stop rules

- The new gate must REFUSE at the round's base commit, and its binding row must
  run the PRODUCTION factory (`make_ocean_physics` / the production momentum
  tendency), never a re-implementation of the operator.  Round 10's D1 defect
  is the reason this is a stop rule and not a preference.
- A one-representable-value perturbation of the production operator's output
  must make the gate refuse and name the cell.
- Each ablation (mask-only, thickness-only) must be shown to MOVE the number;
  an ablation that changes nothing is reported as vacuous, not as agreement.
- The oracle transcription is checked against a case whose answer is already
  known: with a uniform thickness, a unit stretch and an all-ones mask the
  transcription must reproduce legoESM's operator bitwise, so the harness is
  validated before its disagreement is quoted.
- A rigid two-line shift of a round-11 compiled citation must make the citation
  gate FAIL, and the receipt quotes the gate's own fired line (rounds 8 and 9
  passed non-keys to `--plant` and read exit 2 as success; that is withdrawn).
- GYRE's trajectory is proven unchanged before anything lands, at the round's
  base tip and at its final tip, with the evaluation protocol byte-identical.
- No stabiliser, clip, damp or limiter NEMO lacks may be added, and no
  configuration value is chosen that the record does not resolve.

## Labels

The coefficients, metrics, masks and reference thicknesses come from input
files the card already owns, so the operator-gate numbers are **independent**.
The kt=1 velocities and sea surfaces are the record's own recorded frames, so
every number computed on them is **given NEMO's entry**.  The ladder's
trajectory rows keep their existing `INDEPENDENT_WITH_DECISION52_SSH` label.
