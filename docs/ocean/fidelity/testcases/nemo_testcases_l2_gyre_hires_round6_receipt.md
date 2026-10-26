# NEMO testcase L2 GYRE high-resolution — round 6 receipt

Date: 2026-10-13

Incoming lane tip: `255341b45db5536c3dfa9e47b3a3b6143ddc5b6c`

## Verdict

**HELD at the first resolution-rung owner; no physics or deck value changed.**
The requested source-order walk disproves the proposed barotropic-product
grouping fix.  At `r4`, barotropic substep 1 first differs in `slow_u`: 4,158
of 9,520 wet U points differ, the first active point is two ULP away, the
maximum is three ULP and the maximum absolute residue is
`6.617444900424222e-24 m s-2`.  legoESM's production value is bit-identical
to a reconstruction of the written left-associated product from its exposed
operands.  The other two possible three-factor associations are farther from
the NEMO record.  The first owner left open is the surface-stress
transcendental policy upstream of that product.

The round nevertheless lands the requested exact 30-day ladder and its
single-domain acquisition recipe.  It does not run NEMO, MPI, or a GPU.

## Executed source walk

The executed RK3 driver updates a time-varying viscosity, then calls `stp_2D`
with the before level `Nbb` and RHS level `Nrhs` at
`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3.f90:184-190`.  In the external-mode
entry, the vector-form slow RHS is the reference-thickness vertical sum times
the stored reference reciprocal.  The wind is then added as
`r1_rho0 * utauU * (r1_hu_0 / (1 + r3u(Kbb)))`, with the V twin written in the
same order; there is no one-half factor or mask in this statement
(`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stp2d.f90:178-199`).  Thus:

- velocity time level: before, `Nbb`;
- thickness reciprocal time level: reference reciprocal divided by the
  before-level QCO stretch `1+r3(Kbb)`;
- density operand: stored `r1_rho0`;
- no half or mask at the `stp2d` addition.

The half and masks occur upstream when the T-point stresses become face
stresses.  NEMO evaluates `0.5*(tau_i+tau_{i+1})`, then multiplies by
`(2-umask)` and the maximum of the adjacent surface T masks; V uses the
corresponding north pair at
`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/sbcmod.f90:550-559`.  The seasonal clock uses
`REAL(kt)*rn_Dt/(60*60)`, and the first spatial transcendental in the wind
loop is `SIN(pi*(gphit-15)/14)` at
`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/usrdef_sbc.f90:176-191`.

The committed ladder instrument tests the three distinct IEEE programs:
`(rho^-1*tau)*H^-1`, `rho^-1*(tau*H^-1)`, and
`(rho^-1*H^-1)*tau`.  For U, their unequal counts are respectively 4,158,
5,573 and 5,655; none reaches the zero-bit bar.  Production is array-identical
to the first arm.  At rest, the independently built stored reciprocal and the
current `1/H` operand are bit-identical on all active faces, so changing that
operand cannot move substep 1.

The admitted NEMO executable has SHA-256
`974dc326b4f5237a0e8dfe50ba0ae8a4573507d57429fb82069ff3c20abfad41`.
The instrument disassembles only `__usrdef_sbc_MOD_usrdef_sbc_oce` and records
both scalar `sin` and GNU two-lane vector `_ZGVbN2v_sin` calls: the vector call
serves the optimized loop and the scalar call its remainder.  The certified
legoesm policy deliberately evaluates scalar system `sin` once per element.
Together with the eliminated density/depth factors and all product
associations, this localizes the first owner left open to NEMO's vectorized
surface-stress transcendental; the present record does not dump raw `utau`, so
that attribution is an elimination result rather than a direct raw-operand
comparison.  This is a policy/composition difference, not a legal source
regrouping.  Replacing it only for `nn_GYRE=4` would be the forbidden deck
special case.  The round's explicit stop rule therefore fires.

## Ten-step ladders

The production model is unchanged from the incoming tip, so the r4 and
r4_smag whole-step tables reproduce round 5.  Values are maximum absolute
active-field residues against NEMO at each step entry.

| kt | r4 T | r4 S | r4 U | r4 V | r4 SSH | r4_smag T | r4_smag S | r4_smag U | r4_smag V | r4_smag SSH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1.066e-14 | 2.132e-14 | 1.103e-15 | 1.103e-15 | 2.602e-18 | 1.066e-14 | 2.132e-14 | 1.103e-15 | 1.103e-15 | 2.602e-18 |
| 3 | 3.339e-9 | 1.171e-10 | 1.260e-11 | 1.397e-11 | 1.848e-11 | 3.340e-9 | 1.171e-10 | 1.256e-11 | 1.426e-11 | 1.848e-11 |
| 10 | 2.206e-8 | 1.399e-9 | 7.614e-10 | 1.990e-9 | 1.925e-10 | 2.272e-8 | 1.878e-9 | 9.729e-10 | 2.957e-9 | 1.990e-10 |

The corrected Smagorinsky rows remain at the inherited resolution-rung scale,
not the pre-fix `O(1e-2 K)` scale.  Its acquired `ahmt/ahmf` comparison and
the independent NEMO-coefficient operator substitution remain exact/below
the round-5 discriminator bar.

## Smagorinsky branch census and day-30 preregistration

The instrument now evaluates the final `MIN(MAX(deformation, lower), upper)`
owner on NEMO's own before-level velocity at `kt=2`, `3`, and `10`.  The final
measured fractions are:

| kt | T deformation | T lower | T upper | F deformation | F lower | F upper |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 60.4826% | 39.5174% | 0% | 62.4463% | 37.5537% | 0% |
| 3 | 42.8660% | 57.1340% | 0% | 45.1825% | 54.8175% | 0% |
| 10 | 47.6986% | 52.3014% | 0% | 48.6345% | 51.3655% | 0% |

T counts use 288,000 wet cells.  F counts use the 294,030 nonzero mode-32
coefficient points, including free-slip wall points.

Before acquisition, the day-30 prediction is frozen as follows: relative to
`kt=10`, the lower-limit fraction decreases, the deformation-owned fraction
is nonzero, and the upper-limit fraction is nonzero on at least one active
coefficient point.  The prediction may be refuted; it is not a tuning target.
The exact state bar is independently zero unequal active float64 values at
every day boundary under CPU/fp64/scalar-libm policy.

The daily writer records the initial state at `kt=1`, then the state after
each complete day at `kt=25,49,...,721`.  This avoids calling the state before
step 24 an end-of-day state.  The NEMO copy is
`GYRE_OMIP_L2_P3_DAILY`, uses the exact `r4_smag` deck, changes only the run
duration/cadences and a write-only P3 entry condition, and refuses any rank
count other than one because `lwp` records are not gathered.

After acquisition, score it with:

```text
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_30d_ladder_gate.py --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round6/r4_smag_30d_ladder.json
```

## Shipped-deck invariant

No production physics file differs from the incoming tip.  The default card
digest remains `99b2dfce324b0f77`; the ten-step shipped trajectory remains
byte-identical to the registered round-1 legoESM control.  This is the
required no-movement claim, not a claim that every shipped NEMO residual is
zero.  The proposed “old versus NEMO” grouping premise is refuted by the
compiled source: both already evaluate `(r1_rho0*utau)*depth_reciprocal`.
There is consequently no second grouping whose one-degree coincidence needs
an exact-operand explanation.  A vector-transcendental patch would move the
shipped card too, and was not made.

## Tests and evidence

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round6/`.

- focused ladder/acquisition/30-day tests: 25 passed; certified-card digest
  tests: 3 passed;
- daily-writer dry build preflight: PASS;
- r4 and r4_smag ladders: DEBT as preregistered, artifacts written from clean
  commit `8bbe2f2ae` and all one-ULP controls FIRED;
- shipped digest and trajectory no-movement gate: PASS, 70 rows, no worsened
  cell and no legoESM field movement versus the round-1 control;
- receipt citation gate and shifted-citation plant: PASS / FIRED.

## Inherited pins / OPEN

Inherited unchanged are the parameterised shipped, `r4`, and `r4_smag`
namelists; 31 levels; RK3; split-explicit free surface; TKE/EVD; Redi
tracers; seasonal analytic forcing; closed free-slip walls; the exact ladder
bar and one-ULP plants; and the certified CPU fp64/scalar-libm policy.  The
shipped one-degree card remains the default.  The registered post-fix
five-year GPU run launched by the operator remains untouched and must not be
duplicated.

OPEN:

1. The first r4 resolution owner is NEMO's vector-sine wind operand versus
   legoESM's certified scalar-libm operand.  It cannot be removed by a legal
   regrouping of `stp2d` and cannot be special-cased by resolution.
2. The operator must build and run the committed single-rank daily writer;
   then the 30-day exact ladder must score all 31 boundaries.
3. The operator may score the already-running post-fix five-year GPU run when
   it completes; this round launches no replacement.
