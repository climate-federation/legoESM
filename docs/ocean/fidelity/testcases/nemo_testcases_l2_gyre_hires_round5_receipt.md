# NEMO testcase L2 GYRE high-resolution round 5 receipt

**Date:** 2026-10-13
**Incoming tip:** `287abbeb12d2`
**Physics fix:** `b401ec3f3b794b6e4918db372a7cacc802307e15`
**Post-fix run registration:** `72d270a19ff6ac1a599925ee30d81e0041b5caba`
**Review dispositions:** `34609d1eaacfb7ab207d9d5cce33b1c9b5812832`
**Disposition:** **LANDED; post-fix five-year GPU acquisition requested**

## Verdict

The Round-4 operator claim is confirmed and the defect is fixed.  The owner
was the **coefficient**, specifically a second `fmask` multiplication applied
to the completed time-varying mode-32 F-point square-root coefficient.  NEMO
keeps the free-slip perimeter live because its independent velocity lower
limit repopulates those F points; legoESM erased the ring after that limit.
The first port-specific boundary is step 2, RK stage 3, at `dyn_ldf`; U RHS is
the first serialized differing field and V RHS follows in the same record.

With the fix, legoESM's complete dehaloed active-level `ahmt` and `ahmf`
arrays are bit-identical to the new NEMO records at both kt=2 and kt=3.
Substituting the recorded NEMO pair at
the stage-3 `dyn_ldf` boundary leaves only `4.235164736271502e-22 m/s2` maximum
absolute U/V RHS residue, while planting the removed F mask restores a
`1.1498e-8 m/s2` error.  The corrected `r4_smag` kt=3 U/V trajectory errors are
`1.256e-11` and `1.426e-11 m/s`, back at the inherited `r4` scale rather than
the pre-fix `2.99e-5 m/s` scale.

The quarter-degree ladders remain **DEBT** against their deliberately exact
zero-unequal-float bar.  That debt is inherited from the resolution rung; it
is not permission to weaken the bar.  The long-run deliverable remains the
preregistered statistical comparison, and the pre-fix five-year run remains
labelled non-controlling.

## Coefficient-versus-operator discriminator

The admitted AHM files have shape `126 x 86 x 31`, float64 Fortran order, one
domain, and magic `NEMO_L4_AHMTF__1`.  The gate validates the full payload,
dehalos it, maps T/F staggering to the native legoESM arrays, and scores all
30 active levels over the complete native support, including dry/perimeter
points.

| kt | field | values | unequal | max abs | max ULP |
|---:|:---|---:|---:|---:|---:|
| 2 | production `ahmt` | 300120 | 0 | 0 | 0 |
| 2 | production `ahmf` | 306270 | 0 | 0 | 0 |
| 3 | production `ahmt` | 300120 | 0 | 0 | 0 |
| 3 | production `ahmf` | 306270 | 0 | 0 | 0 |
| 2 | planted old-mask `ahmf` | 306270 | 12000 | 255995.3020 | 4688035944438943893 |
| 3 | planted old-mask `ahmf` | 306270 | 12000 | 344134.4978 | 4689655798119550225 |

The production mismatch map is zero in the interior, walls, corners, outside
support, and both clamp classes.  In each planted old-mask arm, all 12,000
differences are on the perimeter over 30 active levels: 11,880 wall values
and 120 corner values, with zero interior/outside differences.  Of those
overlapping classifications, 245 values at kt=2 and 344 at kt=3 carry the
velocity lower clamp; none carries the upper clamp.  NEMO has 294,030 nonzero
F coefficients, while the binary vertex mask has only 282,030 nonzero values.
That exact 12,000-value gap is the erased ring.

The independent operator discriminator changes only the recorded coefficient
pair at the stage-3 call boundary.  Against NEMO's recorded before/after RHS:

| arm | U unequal / max abs / max ULP | V unequal / max abs / max ULP |
|:---|:---|:---|
| NEMO `ahmt/ahmf` | 153 / 4.235e-22 / 2199023255552 | 178 / 4.235e-22 / 549755813888 |
| planted old F mask | 18967 / 1.1498024e-8 / 8946103789027928064 | 16576 / 1.1498012e-8 / 8946101297176360448 |

The large ULP counts in the NEMO-coefficient row occur at values adjacent to
zero; the absolute residue is the controlling disposition and is below
`1e-20 m/s2`.  The planted arm changes the physical increment by fourteen
orders of magnitude more.  This classifies the owner as **COEFFICIENT**, not
the two-pass operator.

The former unit control reconstructed both passes with the production
legoESM Laplacian and was therefore vacuous.  It is replaced by a direct
NumPy transcription of the NEMO div/curl staggering, metrics, masks, e3
weighting, second pass, and negative sign.  A planted legacy F mask must move
that independent result.  A separate acquired-record control proves exact
coefficients without reconstructing NEMO from legoESM's operator.

## Paired quarter-degree ladders

Both gates use CPU fp64/libm, score T/S/U/V/SSH at kt=1 through kt=10, retain
the zero-unequal-float bit bar, and fire one-ULP plants for all five fields.
The table reports maximum absolute error; `r4_smag` is the corrected port.

| deck | kt | T (K) | S | U (m/s) | V (m/s) | SSH (m) |
|:---|---:|---:|---:|---:|---:|---:|
| r4 | 1 | 0 | 0 | 0 | 0 | 0 |
| r4_smag | 1 | 0 | 0 | 0 | 0 | 0 |
| r4 | 2 | 1.066e-14 | 2.132e-14 | 1.103e-15 | 1.103e-15 | 2.602e-18 |
| r4_smag | 2 | 1.066e-14 | 2.132e-14 | 1.103e-15 | 1.103e-15 | 2.602e-18 |
| r4 | 3 | 3.339235e-9 | 1.171117e-10 | 1.259936e-11 | 1.397092e-11 | 1.848316e-11 |
| r4_smag | 3 | 3.339775e-9 | 1.171330e-10 | 1.255830e-11 | 1.426328e-11 | 1.848316e-11 |
| r4 | 10 | 2.206432e-8 | 1.399208e-9 | 7.613925e-10 | 1.990153e-9 | 1.925239e-10 |
| r4_smag | 10 | 2.271991e-8 | 1.878156e-9 | 9.729044e-10 | 2.957170e-9 | 1.990251e-10 |

There is no row that is exact on `r4` but newly non-exact on `r4_smag`.
kt=1 is exact on both, kt=2 is numerically identical field by field, and the
later corrected Smagorinsky rows remain at the same inherited scale while the
two physical decks begin to separate.  The complete artifacts are
`round5/r4_ladder.json` and `round5/r4_smag_ladder.json`.

## Resolution-rung substep diagnosis

The current r4 gate reads `oracle_bt_substeps_*.bin`, not the whole-step
average.  The first differing split-explicit record is substep 1 `slow_u` at
active index `(1,2)`: first difference 2 ULP, maximum 3 ULP, maximum absolute
`6.617444900424222e-24 m/s2`.  Before it, entry/mid SSH and velocity,
continuity, pressure gradient, Coriolis, and rest-state bottom drag are exact.

At the rest start the depth-averaged dynamic RHS is zero.  Replacing only the
frozen NEMO `slow_u/slow_v` composite makes both slow terms, both exit
velocities, and both transport-metric rows bit-exact at substep 1.  The first
owner is therefore upstream of the recurrence in the frozen wind composite
`(r1_rho0 * utau) * r1_hu_0`, not the 50-substep count, its exactly 72-second
step, pressure gradient, drag, or averaging weights.  The record does not
separate `utau`, `r1_rho0`, and `r1_hu_0`, so assigning the two-ULP movement to
one raw multiplicand remains **UNMEASURED**, not guessed.

## Independent-review dispositions and instruments

Two independent adversarial reviews passed after findings were dispositioned.
The physics reviewer checked the compiled source branch, acquired arrays,
Rule-12 override scope, independent two-pass control, JIT execution, and
finite nonzero gradients.  The instrument reviewer first found two real
provenance gaps; the fixes now make the provider refuse:

- a context-window rather than scored-window RMS surface-speed field;
- any non-finite derived movie value on a wet pixel;
- manifest, deck, output cadence, embedded day/step/sample count, file
  sequence, digest, or non-finite-field mutations; and
- a manifest precision policy that differs from the registered `fp64` policy.

The RMS speed map now averages only the scored year, not its 15-day filter
context on each side.  The pre-fix and post-fix registrations pin absolute
run directory, exact commit, deck, mode, backend, precision, transcendental
policy, certification label, and GPU where known.  The post-fix label is
`post_fix_b401ec3f3`; the pre-fix label can never satisfy it.

## Repaired AHM writer recipes

The Round-4 acquisition recipes now match the operator's working repair:
the builder supplies `-d "OCE"`, registers `GYRE_OMIP_L2_P3_AHM OCE`, then
compiles the named config without overwriting its patched `MY_SRC`; the run
tool sources `namelist_ref` and XML files from the P3 parent EXP00; committed
scripts use `grep`; and the `lwp`-guarded, ungathered writer requires exactly
one rank.  Both read-only preflights pass.  Neither `makenemo` nor `mpirun` was
run by this round.

## Tests and evidence

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round5/`.

- corrected `r4_smag` ladder: DEBT only at the inherited resolution scale;
  all five one-ULP controls fired;
- current `r4` ladder: unchanged DEBT; all five controls fired; per-substep
  first owner retained;
- acquired AHM/operator discriminator: coefficient arrays exact, operator
  residue `4.235e-22 m/s2`, planted old mask `1.150e-8 m/s2`;
- affected unit, ladder, acquisition, climate, movie, and non-slow driver
  suites: 35 passed, 2 slow tests deselected;
- certified-card digest plus r4/r4_smag resolution tests: 3 passed; shipped
  GYRE digest remains `99b2dfce324b0f77`;
- physics independent review: PASS, 12 targeted tests plus the slow compiled
  card/JIT/gradient control passed;
- instrument independent review: PASS, 23 focused tests passed and one slow
  test was deselected;
- writer-build and single-rank acquisition preflights: PASS;
- two-day compiled-scan versus per-step CPU control: COMPOSITION debt retained,
  first field SSH at step 3 and maximum surface-T difference
  `1.8106049992638873e-10`; seasonal clock and forcing phase remain bit-equal;
- shipped one-degree ten-step oracle-relative regression: PASS, 70 rows, no
  worsened cell or legoESM movement versus the Round-1 control; and
- receipt citation gate plus shifted-line plant: PASS / FIRED.

## Post-fix acquisition request

GPU 0 was identified by the operator as idle.  The wrapper records that
physical index and refuses a default.  To ensure the registered run's manifest
contains the exact physics SHA rather than a later documentation commit, the
operator should create a detached clean worktree and run:

```text
git -C /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-537618710 worktree add --detach /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/operator-gyrehr-postfix-b401ec3f3 b401ec3f3b794b6e4918db372a7cacc802307e15
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/operator-gyrehr-postfix-b401ec3f3/scripts/run/run_gyre_hires_gpu.sh --physical-gpu 0 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/legoesm_runs/r4_smag_gpu_1825d_post_fix_b401ec3f3
```

Do not kill or displace a running job if GPU 0 is no longer idle.  The output
path is already registered and is the only post-fix legoESM run eligible for
the controlling year-5 comparison.

## Inherited pins / OPEN

Inherited unchanged are the parameterised shipped, `r4`, and `r4_smag`
namelists; 31 levels; RK3; split-explicit free surface; TKE/EVD; Redi
tracers; seasonal analytic forcing; closed free-slip walls; the exact ladder
bar and plants; and the certified CPU fp64/libm policy.  The shipped
one-degree card remains the default with digest `99b2dfce324b0f77` and its
ten-step trajectory is byte-unchanged against the Round-1 control.

For long runs, GPU native transcendentals remain explicitly non-certified.
NEMO daily T/S remain same-file thickness-weighted quantities, the domain is
122 x 82 stitched from eight tiles by `DOMAIN_position_first`, and legoESM
SST daily means use the matching thickness-weighted convention.  The
preregistered year-5 bars remain 20 percent for basin-mean KE and eddy
fraction and two rows for both jet-position measures.  Trajectory divergence
is expected and is not a failure of the climate comparison.

OPEN:

1. the operator must execute the registered post-fix five-year GPU run above;
2. after it completes, the climate and year-1/year-5 movie instruments must
   score the post-fix label against the existing five-year NEMO reference;
3. the raw multiplicand inside the r4 frozen wind composite is unmeasured;
4. the source-composition residue in the bilaplacian operator remains above
   the zero bit bar but is only `4.235e-22 m/s2`; and
5. the later one-twelfth-degree deck and acquisition remain future work.

## NEMO source citations

The static coefficient initializer applies the completed bilaplacian F mask in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:387-393`.  The executed dynamic
mode-32 branch instead constructs its deformation and stability-limited
coefficients and converts them to square-root form in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:462-528`; specifically, its
independent F-point velocity lower limit and final halo exchange, with no
intervening completed-coefficient F mask, are in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:512-534`.

NEMO calls `ldf_dyn` with the before velocity time level in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:159-183`.  The first call that
can consume the new coefficient is the stage-3 lateral momentum update in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:387-398`.

The two level-wise bilaplacian passes use the div/curl, metric, coefficient,
and e3 products in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90:271-290` and
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90:311-330`.

For the independent resolution-rung debt, NEMO forms the vertically averaged
RHS and adds the surface wind forcing in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90:174-199`, then removes the frozen
two-dimensional Coriolis term before the external substep recurrence in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg_ts.f90:284-292`.
