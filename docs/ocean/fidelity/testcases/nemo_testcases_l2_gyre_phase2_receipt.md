# NEMO testcase lane 2 — GYRE phase-2 geometry, forcing, IC, and kt=1 receipt

Date: 2026-09-01

Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Preregistration: `8248a16d3`

Implementation: `bef72a8b0b1ca47917fe827a3e95ea602a0d72e3`

## Verdict

**VERIFIED through the native fp64 `kt=1` Nbb/before entry, and stopped.**
The GYRE card matches all registered oracle geometry at the immutable
pointwise `1e-15` bar, its analytic seasonal SBC matches an independent
source transcription at three phases, and wet T/S/u/v/SSH match the oracle
entry dump exactly.  T/S are informative; at-rest u/v/SSH are explicitly
**UNINFORMATIVE** exact controls.  This is not a trajectory or tendency verdict.

| claim | coverage | result |
|---|---:|---:|
| oracle mesh | 39/39 disposed: 35 VERIFIED, 4 alias WAIVED | 35/35 scored rows AT-BAR; maximum `7.554497157404157e-16` (`e3w_1d`) |
| seasonal SBC | 3 clock samples x 7 fields | 21/21 AT-BAR; maximum `3.3881317890172014e-20` |
| `kt=1` Nbb/before | wet T, S, native u, native v, SSH | T/S exact and informative; u/v/SSH exact but UNINFORMATIVE |

Of the 35 scored geometry rows, 27 independently evaluate card geometry
against the oracle mesh; eight MI96 rows (`e3t/e3w/gdept/gdepw` 1-D plus the
four zco 3-D thickness broadcasts) deliberately compare the phase-1-certified
pinned literals against their source artifact.

Every scientific floating oracle/card array printed by the gate is `float64`.
NEMO masks are int8/float32 and compare exactly to boolean card masks; bottom
and cavity sentinels are integer-exact.  The comparison ran on CPU under
`PrecisionPolicy.fp64()`; the already completed phase-1 NEMO run was not
rerun.

## Base and pre-implementation result

Lane 2 remains based on lane-1 tip
`3d609df4c413d236325d9319f6d93b97dc16fe2a`, because lane 1 was not merged
into `origin/main` when the campaign branch began.  The committed preregister
records the search across the DINO/lane-1 cards, geometry and vertical
constructors, fail-closed time-level registry, gates, and NEMO's running
`usrdef_*` sources.  The existing canonical options covered TEOS-10,
geometric EOS depth, FCT2, the coupled whole-step `rk3_ws` momentum/tracer
identity, UP3, `nemo_sco`, and barotropic composition.  No new solver was
added.

The search also found a real arithmetic constraint: recomputing MI96 with host
NumPy libm misses the oracle by up to `2.386e-15`.  The card therefore pins
the phase-1-certified fp64 arrays emitted by
`usrdef_zgr.F90:93-175` (`nemo_testcase_recipe.py:169-236`) instead of
relaxing the bar.

## Pinned GYRE card

`GYRE-zco` is the closed `32 x 22` beta-plane box with 30 active MI96 levels
and NEMO's dry `jpk=31` record.  It pins the shipped defaults
`rn_Dt=14400`, `nn_itend=4320`, and `nn_GYRE=1`: 4,320 four-hour steps are
720 days, exactly two 360-day years.  The card retains the deliberate
OMIP-style phase-1 EOS deviation `ln_teos10=.true.` and selects the collapsed
campaign identity: `nemo_teos10`, geometric EOS depth, FCT2,
momentum/tracer `rk3_ws`, flux-form UP3, `nemo_up3`, `nemo_sco` plus NEMO
trapezoid, and split-explicit barotropic composition.  The resolved namelist
filter is Demange `nn_bt_flt=3`, alpha `.07`; the live geometry independently
resolves `nn_e=50`, matching `ocean.output:833-839`.  The card and fail-closed
validator are `nemo_testcase_recipe.py:485-648`.

The shipped GYRE oracle uses vector/C2/ENE horizontal momentum whereas the
campaign card deliberately uses lane 1's collapsed flux-form UP3 scheme
identity.  That selector difference is inert at entry.  Its first post-entry
effect and all trajectories are explicitly **UNMEASURED**.

## The two new pieces

The rotated grid is transcribed from `usrdef_hgr.F90` line by line:

- `:75-94` fixes the 106-km spacing, F-point anchor, and `sin/cos(alpha)` for
  the 45-degree rotation;
- `:119-142` supplies the exact T/U/V/F `i-1.5`/`i-1`, `j-1.5`/`j-1`
  staggering;
- `:144-150` makes all eight metrics constant;
- `:158-168` constructs native T/F beta-plane Coriolis.

The card transcription is `nemo_testcase_recipe.py:239-321`.  All eight
coordinates, eight metrics, and two Coriolis arrays are AT-BAR.  The planted
nonzero `glamt + 1` control exits 1 at `0.011672146153025103 > 1e-15`.

The analytic SBC is source-mapped as follows:

- `usrdef_sbc.F90:84-107`: 360-day clock and two seasonal cosines;
- `:109-120`: solar heat flux and restoring target temperature;
- `:122-145`: piecewise freshwater flux and wet mean removal;
- `:161-176`: grid-aligned U/V wind stress from T-point `gphit`;
- `:179-184`: stress and wind magnitudes.

The card evaluator is `nemo_testcase_recipe.py:556-590`; the independent
scalar gate transcription is `nemo_testcase_phase2_gate.py:466-542`.  The
samples are `kt=1` (`t=14400 s`), quarter-year, and half-year.  Quarter-year
has approximately `(cos_sais1, cos_sais2)=(+0.159,-0.356)`, breaking the
degenerate opposite-sign symmetry of the original two samples.
The planted nonzero wet `qsr + 1` control exits 1 at
`0.0060207666701617753 > 1e-15`.

Claude review caught a shared card/gate source misread in the freshwater mean
and correctly required the runtime behavior, not self-agreement, to win.  The
source passes the unmasked 704-cell `emp` array to `glob_2Dsum` and divides by
`glob_2Dsum(tmask)` at `usrdef_sbc.F90:122-140`; the Phase-2 interpretation was
that all 704 numerator cells therefore contributed.  The later write-only
barotropic trace provides the missing runtime discriminator: NEMO's first SSH
increment implies `8.50091456831154e-6`, exactly the numerator over the 600
owned cells, not the naive 704-cell value `9.166209883944753e-6`.  The global
reduction excludes the 104 non-owned boundary-ring cells even though its
argument is syntactically unmasked: `lib_fortran_generic.h90:92,144-148`
multiplies every 2-D operand by `smask0_i`, which `dommsk.F90:200-205` builds
from the unique interior-domain mask.  **Correction:** card and independent
gate now use the owned/wet numerator and denominator; the boundary-ring raw
values remain nonzero and unchanged, while the owned final EMP sum is zero to
fp64 roundoff.  The superseded “land contribution enters the numerator” claim
is retracted rather than silently rewritten.

## Geometry, entry level, controls, and tests

The mesh-driven phase-2 manifest accounts for all 39 file variables.  The four
waivers are only `nav_lon`, `nav_lat`, `nav_lev`, and `time_counter`; their
fp64 scientific coordinate/depth counterparts are verified.  Masks, the
30-level interior census, the no-cavity sentinel, all 31-record MI96 ladders,
and all four 3-D zco thickness fields are scored (`phase2_gate.py:170-330`).

The dump reader calls `time_level_for_dump` and requires **before** before
reading the payload (`phase2_gate.py:375-404`).  Its source is the lane-wide
instrument at
`scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/stprk3.F90:88-100`,
which writes `ts/uu/vv/ssh(...,Nbb)` before forcing, `stp_2D`, or RK stages.
The fp64 header is `(36,26,31)`, then four halos and the dummy bottom record
are explicitly disposed.  All five wet native fields are exact.  The
vertically varying T/S profiles are informative alignment rows; at-rest
u/v/SSH are marked UNINFORMATIVE with reasons, matching lane-1 convention.
The planted wet `T + 1 C` control exits 1 with exact error `inf > 0`.

All four required controls are red:

- file-side unknown: `missing=['PLANTED_UNACCOUNTED_FILE_ARRAY']`;
- rotated `glamt + 1`: normalized error `0.011672146153025103`;
- wet seasonal `qsr + 1`: normalized error `0.0060207666701617753`;
- wet IC `T + 1 C`: exact error `inf`.

The two new-piece controls have direct regression tests.  The combined card,
phase-2 gate, phase-1 time-level, and phase-1 oracle-gate run collected and
passed **68 tests** on CPU.  Ruff passes on all changed Python files, and
`git diff --check` is clean.

## Artifacts and stopping boundary

| artifact | SHA256 |
|---|---|
| oracle mesh | `3bf5d10e36dc52336b9797b13eb1efb4f02d25e0fb3a0c450ac6ce69e65471df` |
| oracle kt=1 entry dump | `9def5f4986853f497a5e0507bea185fe1ec4348715e2aeca0b14507df8e24fa0` |
| phase-2 manifest | `397f0331086e8e6d8d6ee00425418453e1fe46ac401c435487a2f123eba19918` |
| full gate report | `66512c808a75841b09cb4099ba7e8b00b512f0880f09b51c99f4f299822ef116` |
| `usrdef_hgr.F90` / `usrdef_sbc.F90` | `7921aaa7dc9494aa48506f2bd8a6bca3e3952894ef72734fac3e9c050f83011e` / `a2af8adfa5f640947ee3b407138018daa246bc66c6fe887fec7e535bb08fe140` |
| `usrdef_istate.F90` / `usrdef_zgr.F90` | `a597a2add7279a5bdbf1e1292ee02702860e940d2685e743cad3dcac0d878220` / `bfab490058b5a6d685dcccf5e34cea0c71a10cb2533f7c4551fe9939ade40f87` |

The machine receipt keeps `kt>1`, a forced model step, all RK stage
tendencies, TEOS-10 density/`rab`/BN2, FCT and adaptive-vertical-advection
internals, active UP3 momentum parity, barotropic-filter evolution, tracer RK3
parity, BBL transport, and the vector/C2/ENE versus collapsed-UP3 post-entry
comparison loudly **UNMEASURED**.  Phase 2 stops before trajectories.

Phase 1's two prior passes are recorded only as Codex-internal.  Claude's
subsequent Phase-1 verdict was SHIP.  Claude's Phase-2 review also returned
SHIP conditional on the four corrections now recorded here.  The independent
GLM review remains outstanding; Phase 2 makes no dual-review claim.
