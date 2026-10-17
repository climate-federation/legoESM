# NEMO testcase L2 GYRE high resolution — round 8 receipt

Date: 2026-10-13

Incoming lane tip: `63cebeb37ea426cd647688f20e0e0250ae194f3f`

> **Round-9 correction (2026-10-15).**  The shipped-card "trajectory
> regression" discussed below was a cellwise oracle-relative control.  It
> correctly failed because particular corrected cells worsened against NEMO
> by more than its two-row-scale-ULP bar, even though the aggregate row maxima
> moved closer in 37/70 rows, farther in 15, and stayed unchanged in 18.
> Direct before/after numbers are reported in the round-9 receipt.
> Also, the claim below that the record *localized* the next owner to `avt`
> was too strong: the first differing listed boundary only nominated `avt`.
> Round 9 performs the missing one-variable NEMO-`avt` substitution before
> naming that owner.

## Verdict

**HELD at the next measured owner.**  The step-2 tracer owner requested by
this round is real and fixed: the shared physics call deposited the WS-RK3
Kbb two-band solar source with a static attenuation/thickness stretch while
stage 3 subtracted the live-Kbb source.  Passing the live
`1+r3t(Kbb)` stretch removes the material solar-boundary error.  The
one-variable replay falls from `3.2262263971460925e-9 K` with the reverted
static arm to `1.1759482276829658e-12 K` with the production live arm.

The ten-step rung is not closed.  On both quarter-degree decks the first
remaining material temperature error is still step 2 / entry `kt=3`:
`1.81572090696136e-9 K` on `r4` and `1.8162573667268589e-9 K` on
`r4_smag`.  The acquired executed vertical-solve record localizes its first
input boundary to TKE-derived `heat_K` (`avt`), before matrix assembly.  That
is the required stopping point: no TKE hypothesis is promoted and no five-year
GPU rerun is requested while this owner remains.

The shipped one-degree card digest remains `99b2dfce324b0f77`, but its
trajectory is intentionally **not** byte-identical after the transcription
fix.  The live-stretch path becomes active during step 2 there as well.  The
70-row regression has 52 moved rows, all explained below; silently preserving
the old one-degree trajectory would preserve the diagnosed defect.

## Scalar step-2 writer classification

The widened scalar writer is admitted only after a scoped identity proof.
Every byte consumed by the gates is identical to the scalar twin.  Ten
serialized word occurrences at four distinct array indices move outside
NEMO's recorded `ntsi:ntei` / `ntsj:ntej` loop box:

- stage-1 `zFu` at Fortran `[126,60,25]` and `[126,73,13]`;
- stage-2 `zFu` at the same two locations;
- stage-2 `zFw` at `[1,1,1]` and `[2,1,1]`;
- the active-tracer-transport record repeats those two `zFu` and two `zFw`
  locations.

These are uninitialised full-halo work-array words.  The readers crop to the
writer's own loop box before returning the arrays, so earlier gates that
consumed these records never consumed the moving words.  The manifest names
every excluded word.  A one-ULP plant at the first active `zFu` word changes
the scoped verdict from PASS to DEBT, proving that the exception cannot hide
an active-word change.

The acquisition finalizer now supports the operator's already-complete run;
it writes and checks the record manifest and both identity proofs.  The daily
writer builder no longer invokes `dirname` without an operand: it creates the
new config from the registered source config and then copies the selected
`MY_SRC`.  The acquired scalar 30-day root is hash-admitted as
`r4_smag_30d:scalar`; its trajectory score is deferred because the ten-step
TKE owner remains open.

## Step-2 source-order walk

The new writer's prognostic step-1 state and all records the old gates consume
are byte-identical to the scalar twin.  The only raw differences are the ten
excluded work-memory word occurrences above.  The admitted step-2 stream then
gives this temperature walk after bridging NEMO's `kt=2` T/S/U/V/SSH before
state while retaining legoESM's closure memories:

The widened writer's step-1 stage records, step-1 RHS, and `kt=2` entry are
byte-identical to the scalar twin; those are the boundaries earlier rounds
proved exact.  That statement applies only to step 1.  The new step-2 stage
records rotate NEMO's `Kaa` registry slots from `3/2/3` to `1/2/1`, as required
by the compiled RK swaps, and give this measured walk:

| RK stage | T | S | U | V | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.066e-14 | 2.132e-14 | 4.112e-13 | 3.141e-13 | 6.174e-12 |
| 2 | 1.066e-14 | 2.132e-14 | 5.613e-13 | 4.754e-13 | 9.262e-12 |
| 3 | 1.816e-9 | 1.171e-10 | 1.240e-11 | 1.399e-11 | 1.852e-11 |

The first bit difference in source stage/field order is therefore stage 1 T:
`1.0658141036401503e-14 K`, 3 ULP maximum, at `[j=23,i=106,k=6]`.
The first difference at the registered owner-separation scale of `1e-10` is
stage 3 T: `1.81572090696136e-9 K`, 511,080 ULP, at
`[j=1,i=97,k=0]`.  These are measured report fields, not hard-coded labels.

| stage-3 boundary | max absolute T residue | max ULP | location of max `[j,i,k]` |
|---|---:|---:|---:|
| geometry | 1.0302869668521453e-13 | 36 | [1,120,1] |
| advection | 1.1759482276829658e-12 | 1014 | [33,63,16] |
| surface boundary | 1.1759482276829658e-12 | 1014 | [33,63,16] |
| shortwave, fixed live arm | 1.1759482276829658e-12 | 1014 | [33,63,16] |
| lateral diffusion | 1.1741718708435656e-12 | 1014 | [33,63,16] |

The isolated NEMO `tra_qsr` record is bit-identical to the difference between
the two recorded accumulator boundaries: zero unequal words over 288,000 wet
values.  Replaying exactly one changed variable—the Kbb shortwave stretch—
gives:

| arm | max absolute T residue | max ULP | location of max |
|---|---:|---:|---:|
| production live `1+r3t(Kbb)` | 1.1759482276829658e-12 | 1014 | [33,63,16] |
| reverted static one | 3.2262263971460925e-9 | 908102 | [1,120,0] |

The additional independent production-term score compares
`qsr_association.qsr_kmm` directly with NEMO's executed `tra_qsr` record.  Its
maximum residue is `3.5998900258307764e-21 K/s` (17 ULP at that maximum), so it
is a measured floating-point composition debt rather than a second material
QSR transcription error.  This row complements, but does not replace, the
bit-exact NEMO record-versus-accumulator identity above.

Thus the old `3.3392346665550576e-9 K` entry error was not one indivisible
owner.  That old total was 939,911 ULP at `[j=1,i=97,k=0]`; after the QSR fix
the same final-solve location is `1.81572090696136e-9 K`, 511,080 ULP.  The
static Kbb QSR association supplied the material solar-boundary component;
the fixed run exposes a separate vertical owner.

The executed `tra_zdf` matrix/RHS and the production vertical observer agree
on where that next owner first appears:

| vertical operand/boundary | max absolute residue | max ULP | location of max |
|---|---:|---:|---:|
| `heat_K` / `avt` | 5.357229237046984e-10 m2/s | 2662281307 | [27,120,0] |
| isoneutral K | 4.573509118246535e-18 m2/s | 1789670853135789 | [2,119,28] |
| effective K | 5.357229237046984e-10 m2/s | 2662281307 | [27,120,0] |
| Kaa thickness | 1.3073986337985843e-12 m | 37 | [1,118,28] |
| Kmm W thickness | 6.821210263296962e-13 m | 20 | [1,119,23] |
| content RHS | 2.1395862859208137e-10 K m | 941 | [33,63,17] |
| lower diagonal | 1.905514318556456e-7 m | 2510574595 | [27,120,1] |
| main diagonal | 1.9055143241075712e-7 m | 107270920 | [27,120,0] |
| upper diagonal | 1.905514318556456e-7 m | 2510574595 | [27,120,0] |
| solved T | 1.81572090696136e-9 K | 511080 | [1,97,0] |

`heat_K` is the first row in execution/input order, not yet a claim that the
TKE coefficient formula is mistranscribed: this bridge retains legoESM's
hidden TKE memory from step 1.  The next registered experiment is the existing
NEMO-`avt` one-variable substitution at the executed solve boundary; it needs
no new oracle record.  If that confirms coefficient ownership, only then is a
TKE operand writer justified.

## Ten-step scalar ladders

The scalar-libm roots remain controlling.  The exact bar is zero unequal
active float64 words, so both ladders remain DEBT.  All five one-ULP controls
fire on each run.  Values below are maximum absolute entry-state residuals;
the fields are ordered T, S, U, V, SSH.

| kt | r4 T | S | U | V | SSH | r4_smag T | S | U | V | SSH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1.066e-14 | 2.132e-14 | 1.388e-17 | 1.388e-17 | 0 | 1.066e-14 | 2.132e-14 | 1.388e-17 | 1.388e-17 | 0 |
| 3 | 1.816e-09 | 1.171e-10 | 1.260e-11 | 1.397e-11 | 1.848e-11 | 1.816e-09 | 1.171e-10 | 1.256e-11 | 1.426e-11 | 1.848e-11 |
| 4 | 2.599e-09 | 1.375e-10 | 3.344e-11 | 5.517e-11 | 1.701e-11 | 2.758e-09 | 1.459e-10 | 3.332e-11 | 5.886e-11 | 1.705e-11 |
| 5 | 2.652e-09 | 2.622e-10 | 1.110e-10 | 1.679e-10 | 2.455e-11 | 2.789e-09 | 2.668e-10 | 1.251e-10 | 1.965e-10 | 2.438e-11 |
| 6 | 3.671e-09 | 2.135e-10 | 1.415e-10 | 2.344e-10 | 2.609e-11 | 3.861e-09 | 2.152e-10 | 1.709e-10 | 2.805e-10 | 2.613e-11 |
| 7 | 4.410e-09 | 4.049e-10 | 2.703e-10 | 5.677e-10 | 1.737e-11 | 4.704e-09 | 4.310e-10 | 3.483e-10 | 6.426e-10 | 1.756e-11 |
| 8 | 5.181e-09 | 5.665e-10 | 3.130e-10 | 9.285e-10 | 2.323e-11 | 5.782e-09 | 5.879e-10 | 4.932e-10 | 1.043e-09 | 2.433e-11 |
| 9 | 6.073e-09 | 7.386e-10 | 4.241e-10 | 9.459e-10 | 3.431e-11 | 5.132e-09 | 7.434e-10 | 5.516e-10 | 1.030e-09 | 3.487e-11 |
| 10 | 8.384e-09 | 7.107e-10 | 7.589e-10 | 1.696e-09 | 3.552e-11 | 7.634e-09 | 7.585e-10 | 1.319e-09 | 2.560e-09 | 3.643e-11 |

Relative to round 7, plain-r4 T moves from `3.339e-9` to `1.816e-9` at
`kt=3` and from `2.206e-8` to `8.384e-9` at `kt=10`; r4_smag T moves from
`3.340e-9` to `1.816e-9` and from `2.272e-8` to `7.634e-9`.  The nearly equal
`kt=3` rows again separate the common resolution/TKE owner from the
Smagorinsky port.

## Shipped one-degree card

The configuration digest test passes unchanged at `99b2dfce324b0f77`.
That digest covers the resolved card, not the implementation of the shared
shortwave kernel.  The trajectory regression correctly fails because fixing
that shared kernel moves the trajectory.  Rows at `kt=1` and `kt=2` remain
byte-identical; Kbb sea-surface stretch is one at the initial rest state, so
the live and static expressions coincide during step 1.

At `kt=3`, only the direct T row and the two post-step barotropic rows move.
From `kt=4` onward the corrected T has entered density, flow, advection, and
mixing, so all seven certified rows move.  This table enumerates every one of
the 52 changed rows as `moved cells / maximum absolute move from round 7`;
the other 18 certified rows are byte-identical.

| kt | every changed certified row: moved/max move |
|---:|---|
| 3 | after.uu_b: 580/8.534e-13; after.vv_b: 570/1.573e-12; before.T: 8424/1.377e-08 |
| 4 | after.uu_b: 580/2.070e-12; after.vv_b: 570/1.036e-12; before.S: 12292/3.608e-10; before.T: 16373/1.938e-08; before.ssh: 600/1.151e-10; before.u: 17400/1.993e-10; before.v: 17100/1.323e-09 |
| 5 | after.uu_b: 580/4.192e-12; after.vv_b: 570/1.950e-12; before.S: 14504/4.033e-10; before.T: 17000/2.299e-08; before.ssh: 600/1.125e-10; before.u: 17400/4.253e-10; before.v: 17100/5.003e-10 |
| 6 | after.uu_b: 580/5.791e-12; after.vv_b: 570/2.932e-12; before.S: 15344/6.064e-10; before.T: 17267/3.736e-08; before.ssh: 600/2.298e-10; before.u: 17400/7.629e-10; before.v: 17100/9.345e-10 |
| 7 | after.uu_b: 580/8.959e-12; after.vv_b: 570/3.501e-12; before.S: 15677/2.711e-09; before.T: 17396/5.699e-08; before.ssh: 600/3.284e-10; before.u: 17400/1.575e-09; before.v: 17100/5.300e-09 |
| 8 | after.uu_b: 580/1.109e-11; after.vv_b: 570/4.011e-12; before.S: 16105/4.941e-09; before.T: 17485/8.031e-08; before.ssh: 600/4.518e-10; before.u: 17400/1.208e-09; before.v: 17100/4.232e-09 |
| 9 | after.uu_b: 580/1.363e-11; after.vv_b: 570/6.273e-12; before.S: 16357/3.393e-09; before.T: 17536/1.072e-07; before.ssh: 600/6.412e-10; before.u: 17400/1.690e-09; before.v: 17100/4.423e-09 |
| 10 | after.uu_b: 580/1.375e-11; after.vv_b: 570/6.412e-12; before.S: 16459/2.951e-08; before.T: 17603/1.236e-07; before.ssh: 600/8.895e-10; before.u: 17400/1.593e-09; before.v: 17100/2.273e-09 |

This is a deliberate source-fidelity change, not unexplained collateral
movement: all changed rows occur at or downstream of the corrected source,
and no row before that source moves.

## Long-run disposition and inherited pins / OPEN

No new five-year run is registered or requested.  The existing
`pre_fix_7b772afb8` and `post_fix_b401ec3f3` runs remain immutable historical
comparisons; neither contains this QSR correction.  The climate target remains
the preregistered statistical one, including the NEMO-vs-NEMO divergence
envelope, but it is premature to spend another GPU run while a measured
step-2 TKE owner remains.

Inherited unchanged are all deck values, the default shipped one-degree deck,
the `r4` and `r4_smag` namelists, 31 levels, RK3, split-explicit free surface,
Redi tracers, TKE/EVD selection, the seasonal 360-day clock, the scalar-libm
CPU certification policy, the zero-word ladder bar, daily-mean conventions,
and the explicit physical-GPU requirement.  This round changes shared source
arithmetic only; it adds no configuration knob.

OPEN, in order:

1. On the already-acquired scalar step-2 record, substitute NEMO's executed
   `avt` at the vertical-solve boundary.  This is the next one-variable arm
   and requires no acquisition.
2. If that arm reduces solved T to the pre-solve `1.2e-12 K` scale, acquire
   only the TKE operands needed to split input-history from the `tke_avn`
   coefficient statement.  Do not request that writer before the substitution.
3. Only after the TKE owner is closed, run the registered-GPU workflow again
   and score the divergence curve and yearly climate statistics.
4. The scalar 30-day root is admitted but intentionally not certified in this
   held round.

## Tests and evidence

Round-8 evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round8/`.
The controlling artifacts are `step2_tracer_owner.json`,
`r4_scalar_ladder.json`, `r4_smag_scalar_ladder.json`,
`shipped_trajectory.json`, and `shipped_trajectory_regression.json`.

- focused writer/admission tests: 21 passed before the final additions;
- scalar 30-day admission tests: 6 passed;
- climate registry/refusal tests: 10 passed;
- shipped-card digest test: 3 passed;
- final clean lane battery: 121 passed, 2 slow tests deselected;
- compiled-source citation gate: PASS for all four receipt citations; the
  shifted `traqsr` range plant: FIRED.

Two independent reviews were dispositioned before handoff.  The physics
review passed the production fix, its non-vacuous regression, and the
JIT/autodiff checks; its advisory direct-QSR-term row is now in the report.
The evidence review held the first draft because the first RK stage was
inferred and because the identity proof omitted the shared `kt=2` entry and
step-1 RHS.  The measured three-stage walk and the expanded full-byte proof
above close both findings; the active-word plant still refuses.

## Compiled-source citations

The stage-3 tracer call order is surface, two-band solar, lateral diffusion,
then implicit vertical diffusion at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:691-725`.
The executed two-band kernel uses `1+r3t(Kmm)` in both attenuation depth and
layer thickness at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/traqsr.f90:615-645`.
The vertical RHS combines Kbb content and the Kmm stage rate at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/trazdf.f90:284-289`.
The TKE closure computes `avm`/`avt`, including its optional Prandtl update,
at `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdftke.f90:681-692`.
